"""N-Body unified GUI for the CCA.

Choose Sequential / MPI, Bodies, Steps, MPI ranks, then Start.
Exactly the selected number of physics steps executes; timer stops automatically.
Click Metrics / Timing / Workload / Charts for measured information.
MPI is real mpi4py, launched by mpirun inside the background controller.
"""
import csv
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import pygame
from simulation import generate_bodies

BASE = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1280, 780
BG = (12, 17, 29)
CARD = (20, 28, 44)
EDGE = (40, 53, 74)
TEXT = (231, 238, 248)
MUTED = (143, 160, 184)
ACCENT = (102, 224, 194)
BLUE = (113, 164, 255)
AMBER = (255, 200, 115)
RED = (239, 122, 120)
BODY_COUNTS = (10, 50, 100, 103, 500, 1000, 5000)
STEP_COUNTS = (10, 50, 100, 500, 1000)
RANK_COUNTS = (1, 2, 4, 8)


class Session:
    """One background worker; no physics calculation runs on the GUI thread."""

    def __init__(self, events, ident, mode, bodies, ranks, steps):
        self.events = events
        self.ident = ident
        self.mode, self.bodies, self.ranks, self.steps = mode, bodies, ranks, steps
        self.stopping = threading.Event()
        self.proc = None
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def emit(self, message):
        self.events.put((self.ident, message))

    def shutdown(self):
        self.stopping.set()
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                if os.name == 'posix':
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            except (OSError, ProcessLookupError):
                pass

    def run(self):
        command = [sys.executable, '-u', str(BASE / 'integrated_worker.py'),
                   '--mode', self.mode, '--bodies', str(self.bodies),
                   '--seed', '42', '--dt', '0.02',
                   '--preview-every', str(max(1, math.ceil(self.steps / 50)))]
        if self.mode == 'mpi':
            command = ['mpirun', '--oversubscribe', '-np', str(self.ranks)] + command
        try:
            proc = subprocess.Popen(command, cwd=BASE, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    bufsize=1, text=True, start_new_session=(os.name == 'posix'))
            self.proc = proc
            completed = False
            for line in proc.stdout:
                if self.stopping.is_set():
                    break
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # Some MPI installations can print runtime diagnostics.
                    self.emit({'type': 'info', 'message': line[:180].strip()})
                    continue
                kind = message.get('type')
                if kind == 'ready':
                    self.emit(message)
                    proc.stdin.write(f'RUN {self.steps}\n')
                    proc.stdin.flush()
                elif kind in ('started', 'progress', 'finished'):
                    self.emit(message)
                    if kind == 'finished':
                        completed = True
                        break
            if not self.stopping.is_set():
                proc.wait(timeout=15)
                if proc.returncode:
                    error = proc.stderr.read()[-1400:]
                    raise RuntimeError(error or f'Worker exited with code {proc.returncode}')
                if not completed:
                    raise RuntimeError('Worker ended without a completed result')
        except Exception as exc:
            if not self.stopping.is_set():
                self.emit({'type': 'error', 'message': str(exc)})
        finally:
            self.shutdown()
            proc = self.proc
            if proc is not None:
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try: proc.kill()
                    except OSError: pass
                    proc.wait()
            self.proc = None


class App:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('N-Body Simulation | Sequential + MPI | CCA')
        self.clock = pygame.time.Clock()
        self.fonts = {n: pygame.font.SysFont('segoeui', n) for n in (12, 13, 14, 15, 16, 18, 22, 28)}
        self.mode = 'sequential'
        self.bodies = 100
        self.ranks = 4
        self.steps_limit = 100
        self.tab = 'Metrics'
        self.state = 'READY'
        self.events = queue.SimpleQueue()
        self.session = None
        self.session_id = 0
        self.buttons = {}
        self.comparison = []
        self.latest = {}
        self.trails = []
        self.positions = []
        self.colors = []
        self.selected = None
        self.live_anchor = None
        self.message = 'Choose bodies and steps, then click Start. The run stops automatically.'
        self._reset_display()
        rng = random.Random(19)
        self.stars = [(rng.randrange(820), rng.randrange(480),
                       rng.choice(((36, 48, 68), (58, 72, 94)))) for _ in range(150)]

    def _reset_display(self):
        initial = generate_bodies(self.bodies, 42)
        self.positions = [[b[0], b[1]] for b in initial]
        self.colors = [b[5] for b in initial]
        self.trails = [deque(maxlen=85) for _ in initial]
        self.latest = {'steps': 0, 'elapsed': 0.0, 'compute': 0.0,
                       'communication': 0.0, 'processes': 1 if self.mode=='sequential' else self.ranks,
                       'counts': [self.bodies] if self.mode=='sequential' else []}
        self.live_anchor = None
        self.selected = None

    def stop(self, manual=False):
        # Freeze the visible live estimate if the user cancels mid-run.
        # An interrupted run is never used for CCA speedup or saved as complete.
        partial_elapsed = self.displayed_elapsed() if manual else None
        if self.session:
            self.session.shutdown()
        self.session = None
        self.session_id += 1  # Ignore events from a cancelled process.
        self.live_anchor = None
        if manual:
            if partial_elapsed is not None:
                self.latest['elapsed'] = partial_elapsed
            self.state = 'STOPPED'
            self.message = 'Stopped early. This incomplete run is NOT used for speedup.'

    def start(self):
        self.stop()
        self._reset_display()
        self.state = 'STARTING'
        self.message = 'Initializing bodies and processes; timing begins with physics.'
        self.session = Session(self.events, self.session_id, self.mode,
                               self.bodies, self.ranks, self.steps_limit)

    def save_result(self, result):
        row = {key: result[key] for key in
               ('mode', 'bodies', 'steps', 'processes', 'seed', 'dt',
                'elapsed', 'compute', 'communication')}
        self.comparison = [old for old in self.comparison if not all(
            old[k] == row[k] for k in ('mode', 'bodies', 'steps', 'processes'))]
        self.comparison.append(row)
        folder = BASE / 'benchmarking'
        folder.mkdir(exist_ok=True)
        filename = folder / 'gui_runs.csv'
        columns = ['mode','bodies','steps','processes','seed','dt',
                   'elapsed','compute','communication']
        with filename.open('a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if f.tell() == 0: writer.writeheader()
            writer.writerow(row)

    def handle(self, data):
        kind = data.get('type')
        if kind == 'ready':
            self.colors = data['colors']
            self.trails = [deque(maxlen=85) for _ in self.colors]
            self.latest['processes'] = data['processes']
            self.latest['counts'] = data['counts']
            self.message = f'Initialized {self.bodies} bodies, {data["processes"]} process(es).'
        elif kind == 'started':
            self.state = 'RUNNING'
            self.live_anchor = time.perf_counter()
            self.message = f'Calculating exactly {self.steps_limit} steps...'
        elif kind == 'progress':
            self.latest.update({k: data[k] for k in ('steps', 'elapsed', 'compute', 'communication')})
            self.positions = data['positions']
            for trail, pos in zip(self.trails, self.positions):
                trail.append(tuple(pos))
        elif kind == 'finished':
            self.latest.update({k: data[k] for k in
                  ('steps', 'elapsed', 'compute', 'communication', 'counts', 'processes')})
            self.positions = data['positions']
            self.live_anchor = None
            self.state = 'FINISHED'
            self.message = (f'Completed {data["steps"]} steps. Final execution time: '
                            f'{data["elapsed"]:.6f} seconds.')
            self.save_result(data)
        elif kind == 'info':
            self.message = data['message']
        elif kind == 'error':
            self.live_anchor = None
            self.state = 'ERROR'
            self.message = data['message'][:175]

    def displayed_elapsed(self):
        if self.state == 'RUNNING' and self.live_anchor is not None:
            # Smooth live estimate. The completed value comes from the
            # physics worker's one start/end clock, not the GUI clock.
            return max(self.latest['elapsed'], time.perf_counter() - self.live_anchor)
        return self.latest['elapsed']

    def text(self, s, x, y, size=14, color=TEXT):
        self.screen.blit(self.fonts[size].render(str(s), True, color), (int(x), int(y)))

    def box(self, rect, color=CARD):
        pygame.draw.rect(self.screen, color, rect, border_radius=12)
        pygame.draw.rect(self.screen, EDGE, rect, 1, border_radius=12)

    def button(self, key, label, rect, active=False, primary=False, disabled=False):
        self.buttons[key] = (rect, disabled)
        hovered = rect.collidepoint(pygame.mouse.get_pos())
        color = (42, 74, 77) if active else ((31, 45, 65) if hovered else (27, 38, 55))
        if primary: color = ACCENT
        if disabled: color = (25, 32, 44)
        pygame.draw.rect(self.screen, color, rect, border_radius=7)
        pygame.draw.rect(self.screen, ACCENT if active else EDGE, rect, 1, border_radius=7)
        label_color = MUTED if disabled else (BG if primary else TEXT)
        surf = self.fonts[14].render(label, True, label_color)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def draw_config(self):
        self.box(pygame.Rect(24, 82, 1232, 91))
        busy = self.state in ('STARTING', 'RUNNING')
        self.text('BODIES', 42, 92, 12, MUTED)
        for i, n in enumerate(BODY_COUNTS):
            self.button(('bodies', n), str(n), pygame.Rect(42+i*63, 113, 58, 36),
                        active=self.bodies==n, disabled=busy)
        self.text('STEPS', 507, 92, 12, MUTED)
        for i, n in enumerate(STEP_COUNTS):
            self.button(('steps', n), str(n), pygame.Rect(507+i*60,113,55,36),
                        active=self.steps_limit==n, disabled=busy)
        self.text('MPI PROCESSES', 829, 92, 12, MUTED)
        for i, n in enumerate(RANK_COUNTS):
            self.button(('rank', n), str(n), pygame.Rect(829+i*53,113,47,36),
                        active=self.ranks==n, disabled=busy or self.mode!='mpi')
        self.button('start','Start',pygame.Rect(1053,113,86,36),primary=True,disabled=busy)
        self.button('stop','Stop',pygame.Rect(1148,113,89,36),disabled=not busy)

    def draw_space(self):
        canvas = pygame.Rect(24, 190, 824, 483)
        self.box(canvas, (8, 13, 25))
        prev = self.screen.get_clip(); self.screen.set_clip(canvas)
        for x,y,color in self.stars:
            pygame.draw.circle(self.screen, color, (canvas.x+x, canvas.y+y), 1)
        scale=min((canvas.w-50)/800.,(canvas.h-50)/600.)
        ox,oy=canvas.centerx-400*scale,canvas.centery-300*scale
        def point(pos):return (int(ox+pos[0]*scale),int(oy+pos[1]*scale))
        for color,trail in zip(self.colors,self.trails):
            if len(trail)>1:
                pygame.draw.lines(self.screen, tuple(int(c*.55) for c in color),
                                  False, [point(p) for p in trail], 1)
        for i,(pos,color) in enumerate(zip(self.positions,self.colors)):
            xy=point(pos)
            if canvas.collidepoint(xy):
                pygame.draw.circle(self.screen,color,xy,3 if self.bodies>250 else 4)
                if i==self.selected:pygame.draw.circle(self.screen,TEXT,xy,10,1)
        self.screen.set_clip(prev)
        self.text('LIVE GRAVITY SIMULATION',44,204,13,MUTED)
        self.text('Same seed 42  |  dt = 0.02  |  fixed number of physics steps',44,646,12,MUTED)

    def meter(self,label,value,x,y,color=ACCENT):
        self.text(label,x,y,13,MUTED)
        self.text(value,x,y+22,22,color)

    def matching_results(self):
        return sorted((r for r in self.comparison if
                       r['bodies']==self.bodies and r['steps']==self.steps_limit),
                      key=lambda r:(r['mode']!='sequential',r['processes']))

    def draw_metrics(self):
        self.box(pygame.Rect(864,190,392,483))
        for i,tab in enumerate(('Metrics','Timing','Workload','Charts')):
            self.button(('tab',tab),tab,pygame.Rect(878+i*91,205,86,34),active=self.tab==tab)
        x,y=885,261
        steps=self.latest['steps']; elapsed=self.displayed_elapsed()
        comp=self.latest['compute']; comm=self.latest['communication']
        mpi=self.mode=='mpi'
        if self.tab=='Metrics':
            self.text('Current experiment',x,y,18)
            self.meter('Mode', 'MPI' if mpi else 'SEQUENTIAL',x,y+38,BLUE)
            self.meter('Bodies',str(self.bodies),x+202,y+38)
            self.meter('Processes',str(self.latest['processes'] if mpi else 1),x,y+116)
            self.meter('Steps completed',f'{steps} / {self.steps_limit}',x+202,y+116)
            self.meter('TOTAL EXECUTION TIME',f'{elapsed:.6f} s',x,y+208)
            self.text('Computer clock: start of physics to final step',x,y+284,12,MUTED)
            self.text('Final time freezes automatically at completion.',x,y+306,12,MUTED)
            self.text('Start a matching MPI/Sequential run for comparison.',x,y+343,12,MUTED)
        elif self.tab=='Timing':
            self.text('Actual measured execution time',x,y,18)
            self.meter('TOTAL',f'{elapsed:.6f} s',x,y+35)
            self.text(f'{elapsed*1000:.3f} milliseconds',x,y+110,15,BLUE)
            self.text(f'Calculation (diagnostic): {comp:.6f} s',x,y+153,14)
            self.text(f'MPI communication: {comm:.6f} s' if mpi else
                      'Communication: N/A (Sequential)',x,y+187,14)
            self.text(f'Completed steps: {steps} / {self.steps_limit}',x,y+227,14)
            self.text('One start/end timer covers the full physics loop.',x,y+273,12,MUTED)
            self.text('MPI total includes its inter-process communication.',x,y+299,12,MUTED)
            self.text('Diagnostic maxima need not add up to total.',x,y+323,12,MUTED)
        elif self.tab=='Workload':
            self.text('Distribution of bodies',x,y,18)
            counts=self.latest.get('counts') or ([self.bodies] if not mpi else
                    [self.bodies//self.ranks+(i<self.bodies%self.ranks)
                     for i in range(self.ranks)])
            for i,n in enumerate(counts):
                yy=y+45+i*38
                self.text(f'Rank {i}',x,yy,14)
                pygame.draw.rect(self.screen,EDGE,pygame.Rect(x+77,yy+3,168,12),border_radius=5)
                pygame.draw.rect(self.screen,BLUE,pygame.Rect(x+77,yy+3,
                                 max(2,int(168*n/max(counts))),12),border_radius=5)
                self.text(f'{n} bodies',x+256,yy,14)
            self.text(f'Total: {sum(counts)} / {self.bodies}',x,y+377,13,MUTED)
        else:
            self.text('Your measured runs (same bodies + steps)',x,y,18)
            matches=self.matching_results()
            if not matches:
                self.text('Run Sequential, then MPI using the same',x,y+65,14,MUTED)
                self.text('body and step counts. Results appear here.',x,y+88,14,MUTED)
                return
            max_time=max(r['elapsed'] for r in matches) or 1.
            baseline=next((r['elapsed'] for r in matches if r['mode']=='sequential'),None)
            for i,r in enumerate(matches[:5]):
                yy=y+41+i*63
                label='Sequential' if r['mode']=='sequential' else f'MPI {r["processes"]} processes'
                self.text(f'{label}: {r["elapsed"]:.6f} s',x,yy,14)
                pygame.draw.rect(self.screen,EDGE,pygame.Rect(x,yy+24,329,12),border_radius=5)
                pygame.draw.rect(self.screen,ACCENT if r['mode']=='sequential' else BLUE,
                                 pygame.Rect(x,yy+24,max(2,int(329*r['elapsed']/max_time)),12),
                                 border_radius=5)
                if baseline is not None and r['mode']=='mpi':
                    speed=baseline/r['elapsed'] if r['elapsed'] else 0
                    self.text(f'Speedup {speed:.2f}x  |  Efficiency {100*speed/r["processes"]:.1f}%',
                              x+5,yy+40,12,MUTED)
            self.text('Same workload, seed 42, dt = 0.02.',x,y+365,12,MUTED)

    def draw(self):
        self.buttons={}
        self.screen.fill(BG)
        self.text('N-Body Simulation Lab',24,15,28)
        self.text('Distributed Computing | Sequential vs Real MPI',25,54,14,MUTED)
        busy=self.state in ('STARTING','RUNNING')
        self.button(('mode','sequential'),'SEQUENTIAL',pygame.Rect(726,23,150,40),
                    active=self.mode=='sequential',disabled=busy)
        self.button(('mode','mpi'),'MPI DISTRIBUTED',pygame.Rect(887,23,165,40),
                    active=self.mode=='mpi',disabled=busy)
        self.text(self.state,1073,36,16,
                  ACCENT if self.state in ('RUNNING','FINISHED') else
                  RED if self.state=='ERROR' else AMBER)
        self.draw_config()
        self.draw_space()
        self.draw_metrics()
        self.text('Select bodies + steps, click Start, wait for FINISHED. Switch mode and repeat.',
                  24,696,14,MUTED)
        self.text('Timing: one start-to-end computer clock; GUI drawing and MPI startup excluded.',
                  24,722,13,MUTED)
        self.text(self.message[:190],24,751,13,RED if self.state=='ERROR' else MUTED)
        pygame.display.flip()

    def click(self,position):
        busy=self.state in ('RUNNING','STARTING')
        for key,(rect,disabled) in self.buttons.items():
            if disabled or not rect.collidepoint(position):
                continue
            if isinstance(key,tuple):
                kind,value=key
                if kind=='tab':
                    self.tab=value
                elif kind=='mode' and not busy:
                    if self.mode!=value:
                        self.mode=value; self._reset_display(); self.state='READY'
                elif kind=='bodies' and not busy:
                    if self.bodies!=value:
                        self.bodies=value; self._reset_display(); self.state='READY'
                elif kind=='steps' and not busy:
                    self.steps_limit=value; self.state='READY'
                    self._reset_display()
                elif kind=='rank' and self.mode=='mpi' and not busy:
                    self.ranks=value; self._reset_display(); self.state='READY'
            elif key=='start':
                self.start()
            elif key=='stop':
                self.stop(manual=True)
            return

    def run(self):
        open_ = True
        while open_:
            self.clock.tick(60)
            for event in pygame.event.get():
                if event.type==pygame.QUIT or (
                     event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE):
                    open_=False
                elif event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                    self.click(event.pos)
            for _ in range(32):
                try: ident,message=self.events.get_nowait()
                except queue.Empty: break
                if ident==self.session_id:
                    self.handle(message)
            self.draw()
        self.stop()
        pygame.quit()


if __name__=='__main__':
    App().run()
