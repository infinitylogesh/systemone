"""Maze and Snake for the arcade demo: boards, prompts, rendering and baselines.

Shared by the streaming endpoint (`systemone serve --demo`, POST /arcade/play) and
arcade_eval.py, so both play exactly the same games. web/arcade.html has a
JavaScript copy for its browser-driven mode.
"""

import base64
import struct
import zlib

DIRS = {"north": (-1, 0), "east": (0, 1), "south": (1, 0), "west": (0, -1)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}
MOVE_LABEL = {
    "north": "move up one row",
    "east": "move right one column",
    "south": "move down one row",
    "west": "move left one column",
}
LAYOUT = (
    "cached"  # "legacy": per-move details in the options (the first arcade version)
)


def finish(body, lines, opts):
    """Place the per-move details: in the state (cached layout) or as option descriptions (legacy)."""
    q = body["questions"]["move"]
    if LAYOUT == "legacy":
        q["criteria"] = {d: line.split(": ", 1)[1] for d, line in zip(opts, lines)}
        body["state"] = (
            body["state"]
            .rsplit("\nMoves you can make now:", 1)[0]
            .rsplit("\nSafe moves now:", 1)[0]
        )
    return body


def png(cells, px=16):
    """cells: 2-D list of RGB tuples -> PNG data URL."""
    h, w = len(cells), len(cells[0])
    rows = []
    for r in range(h * px):
        row = bytearray(b"\x00")
        for c in range(w * px):
            row += bytes(cells[r // px][c // px])
        rows.append(bytes(row))

    def ch(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d))

    data = b"\x89PNG\r\n\x1a\n" + ch(
        b"IHDR", struct.pack(">IIBBBBB", w * px, h * px, 8, 2, 0, 0, 0)
    )
    data += ch(b"IDAT", zlib.compress(b"".join(rows))) + ch(b"IEND", b"")
    return "data:image/png;base64," + base64.b64encode(data).decode()


WALL, FLOOR, SEEN, AGENT, GOAL, SNAKE, HEAD, FOOD = (
    (43, 42, 39),
    (251, 250, 247),
    (239, 227, 200),
    (47, 111, 221),
    (31, 157, 85),
    (47, 111, 221),
    (27, 79, 174),
    (217, 72, 15),
)


class Maze:
    def __init__(self, n, rng):
        self.n, H = n, 2 * n + 1
        self.g = [["#"] * H for _ in range(H)]
        seen, stack = {(0, 0)}, [(0, 0)]
        self.g[1][1] = "."
        while stack:
            r, c = stack[-1]
            nb = [
                (r + a, c + b)
                for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0))
                if 0 <= r + a < n and 0 <= c + b < n and (r + a, c + b) not in seen
            ]
            if not nb:
                stack.pop()
                continue
            a, b = rng.choice(nb)
            self.g[r + a + 1][c + b + 1] = "."
            self.g[2 * a + 1][2 * b + 1] = "."
            seen.add((a, b))
            stack.append((a, b))
        self.pos, self.goal = (1, 1), (H - 2, H - 2)
        self.visits = {self.pos: 1}
        self.moves, self.limit, self.done, self.won = 0, n * n * 6, False, False

    def legal(self):
        return [
            d
            for d, (a, b) in DIRS.items()
            if self.g[self.pos[0] + a][self.pos[1] + b] != "#"
        ]

    def move(self, d):
        self.pos = (self.pos[0] + DIRS[d][0], self.pos[1] + DIRS[d][1])
        self.moves += 1
        self.visits[self.pos] = self.visits.get(self.pos, 0) + 1
        if self.pos == self.goal:
            self.done = self.won = True
        elif self.moves >= self.limit:
            self.done = True

    def ascii(self, marks):
        out = []
        for r, row in enumerate(self.g):
            s = ""
            for c, ch in enumerate(row):
                s += (
                    "@"
                    if (r, c) == self.pos
                    else "G"
                    if (r, c) == self.goal
                    else "v"
                    if marks and ch == "." and (r, c) in self.visits
                    else ch
                )
            out.append(s)
        return "\n".join(out)

    def image(self, hints):
        cells = [
            [
                WALL
                if ch == "#"
                else (SEEN if hints and (r, c) in self.visits else FLOOR)
                for c, ch in enumerate(row)
            ]
            for r, row in enumerate(self.g)
        ]
        cells[self.goal[0]][self.goal[1]] = GOAL
        cells[self.pos[0]][self.pos[1]] = AGENT
        return png(cells)

    def request(self, mode, hints):
        (r, c), (gr, gc) = self.pos, self.goal
        dr, dc = gr - r, gc - c
        where = (
            f"You are at row {r}, column {c}. The goal is at row {gr}, column {gc} ({abs(dr)} rows {'down' if dr >= 0 else 'up'}, "
            f"{abs(dc)} columns {'right' if dc >= 0 else 'left'})."
        )
        key = (
            "# wall, . open floor, v open floor you already visited, @ you, G goal."
            if hints
            else "# wall, . open floor, @ you, G goal."
        )
        if mode == "image":
            state = (
                "The image shows a maze seen from above: dark squares are walls, light squares are floor, the blue circle is you "
                "and the green square is the goal"
                + (
                    "; tinted floor squares are cells you already visited"
                    if hints
                    else ""
                )
                + f". Row 0 is the top edge, column 0 the left edge. {where}"
            )
        else:
            state = (
                f"A maze seen from above. Row 0 is the top, column 0 the left. {key}\n{self.ascii(hints)}\n{where}"
                + (
                    "\nThe image shows the same maze (blue circle = you, green square = goal)."
                    if mode == "both"
                    else ""
                )
            )
        lines = []
        for d in self.legal():
            p = (r + DIRS[d][0], c + DIRS[d][1])
            v = self.visits.get(p, 0)
            lines.append(
                f"- {d}: to row {p[0]}, column {p[1]}"
                + (
                    " (the goal)"
                    if p == self.goal
                    else (
                        (
                            f", already visited {v} time{'s' if v > 1 else ''}"
                            if v
                            else ", not visited yet"
                        )
                        if hints
                        else ""
                    )
                )
            )
        state += "\nMoves you can make now:\n" + "\n".join(lines)
        body = {
            "state": state,
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Which move should you make next to reach the goal G? Walls block movement. Explore cells you have not visited, "
                    "and back out of dead ends.",
                    "criteria": {d: MOVE_LABEL[d] for d in self.legal()},
                }
            },
        }
        if mode != "text":
            body["images"] = [self.image(hints)]
        return finish(body, lines, self.legal())

    def heuristic(self, rng):
        opts = self.legal()
        best = min(
            self.visits.get((self.pos[0] + DIRS[d][0], self.pos[1] + DIRS[d][1]), 0)
            for d in opts
        )
        return rng.choice(
            [
                d
                for d in opts
                if self.visits.get(
                    (self.pos[0] + DIRS[d][0], self.pos[1] + DIRS[d][1]), 0
                )
                == best
            ]
        )


class Snake:
    def __init__(self, n, rng):
        self.n, self.rng, m = n, rng, n // 2
        self.body, self.dir = [(m, 2), (m, 1), (m, 0)], "east"
        self.score = self.moves = self.hunger = 0
        self.done = self.won = False
        self.why = ""
        self.place()

    def place(self):
        free = [
            (r, c)
            for r in range(self.n)
            for c in range(self.n)
            if (r, c) not in self.body
        ]
        if not free:
            self.done = self.won = True
            self.why = "board full"
            return
        self.food = self.rng.choice(free)

    def nxt(self, d):
        return (self.body[0][0] + DIRS[d][0], self.body[0][1] + DIRS[d][1])

    def safe(self):
        out = []
        for d in DIRS:
            if d == OPP[self.dir]:
                continue
            p = self.nxt(d)
            body = self.body if p == self.food else self.body[:-1]
            if 0 <= p[0] < self.n and 0 <= p[1] < self.n and p not in body:
                out.append(d)
        return out

    def space(self, d):
        p = self.nxt(d)
        body = [p] + (self.body if p == self.food else self.body[:-1])
        blocked, seen, q = set(body), {p}, [p]
        while q:
            r, c = q.pop()
            for a, b in DIRS.values():
                k = (r + a, c + b)
                if (
                    0 <= k[0] < self.n
                    and 0 <= k[1] < self.n
                    and k not in blocked
                    and k not in seen
                ):
                    seen.add(k)
                    q.append(k)
        return len(seen) - 1

    def move(self, d):
        p = self.nxt(d)
        self.dir = d
        self.moves += 1
        self.hunger += 1
        self.body.insert(0, p)
        if p == self.food:
            self.score += 1
            self.hunger = 0
            self.place()
        else:
            self.body.pop()
        if self.hunger > self.n * self.n * 2:
            self.done, self.why = True, "went in circles"

    def ascii(self):
        g = [["."] * self.n for _ in range(self.n)]
        for i, (r, c) in enumerate(self.body):
            g[r][c] = "H" if i == 0 else ("T" if i == len(self.body) - 1 else "o")
        g[self.food[0]][self.food[1]] = "F"
        return "\n".join("".join(r) for r in g)

    def image(self):
        cells = [[FLOOR] * self.n for _ in range(self.n)]
        cells[self.food[0]][self.food[1]] = FOOD
        for i, (r, c) in enumerate(self.body):
            cells[r][c] = HEAD if i == 0 else SNAKE
        return png(cells, 24)

    def request(self, mode, hints):
        (r, c), (fr, fc) = self.body[0], self.food
        dr, dc = fr - r, fc - c
        where = (
            f"The head is at row {r}, column {c}, moving {self.dir}. The food is at row {fr}, column {fc} ({abs(dr)} rows "
            f"{'down' if dr >= 0 else 'up'}, {abs(dc)} columns {'right' if dc >= 0 else 'left'}). Length {len(self.body)}."
        )
        if mode == "image":
            state = (
                f"The image shows a Snake game on a {self.n}x{self.n} board: the blue squares are the snake (the dark blue one is "
                f"its head), the orange circle is the food. Row 0 is the top, column 0 the left. {where}"
            )
        else:
            state = (
                f"A Snake game on a {self.n}x{self.n} board. Row 0 is the top, column 0 the left. . empty, H head, o body, T tail, "
                f"F food. Hitting a wall or the body ends the game.\n{self.ascii()}\n{where}"
            ) + ("\nThe image shows the same board." if mode == "both" else "")
        lines = []
        for d in self.safe():
            p = self.nxt(d)
            desc = f"- {d}: to row {p[0]}, column {p[1]}"
            if p == self.food:
                desc += " (eats the food)"
            elif hints:
                before, after = (
                    abs(fr - r) + abs(fc - c),
                    abs(fr - p[0]) + abs(fc - p[1]),
                )
                desc += (
                    ", closer to the food" if after < before else ", away from the food"
                )
                desc += f", leaves {self.space(d)} free cells reachable"
            lines.append(desc)
        state += "\nSafe moves now:\n" + "\n".join(lines)
        body = {
            "state": state,
            "questions": {
                "move": {
                    "type": "choice",
                    "instructions": "Which way should the snake move next? Reach the food without trapping the snake: never move into a pocket too "
                    "small for its body.",
                    "criteria": {d: MOVE_LABEL[d] for d in self.safe()},
                }
            },
        }
        if mode != "text":
            body["images"] = [self.image()]
        return finish(body, lines, self.safe())

    def heuristic(self, rng):
        opts = self.safe()
        need = len(self.body)

        def key(d):
            p = self.nxt(d)
            dist = abs(self.food[0] - p[0]) + abs(self.food[1] - p[1])
            return (self.space(d) < need, dist, -self.space(d))

        return min(opts, key=key)
