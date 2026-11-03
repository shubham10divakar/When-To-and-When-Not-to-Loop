"""Pointer-chase task generator (design doc §5.1).

A task is an in-memory directory of files. The true chain ``chain[0] → … →
chain[n-1]`` has ``chain[0]`` named in the prompt; each chain file holds one
valid clue (tagged with the task key) identifying the next file, and the last
holds ``[KEY] ANSWER: <token>``. Reaching the answer takes exactly ``n``
correct reads (n = depth).

Decoy chains end in fake answers and are reachable only through a wrong read or
a silent injected failure (which rewrites the valid clue to point into a decoy
chain at the same depth). Every file is padded with seeded filler to an *exact*
token count under :func:`~loopbounds.harness.tokens.count_tokens`.

Difficulty levels:
  L0  clue states the file name directly
  L1  L0 + fake clues carrying other keys
  L2  L1 + indirection ("the file whose name, reversed, is …")
  L3  L2 + the clue is split across two lines separated by filler
"""

from __future__ import annotations

import hashlib
import json
import string
from dataclasses import dataclass, field, asdict

import numpy as np

from loopbounds.harness.tokens import CHARS_PER_TOKEN

DIFFICULTIES = ("L0", "L1", "L2", "L3")
NAME_ALPHABET = string.ascii_lowercase + string.digits
TOKEN_ALPHABET = string.ascii_uppercase + string.digits

# A fixed, contentless vocabulary for filler text (seeded shuffles give
# natural-looking but meaningless prose).
_WORDS = (
    "the of and a to in is was that for on with as by at from it his an were are which this be or "
    "had not but have one their all there been has when who will more no if out so said what up "
    "its about into than them can only other new some could time these two may then do first any "
    "like now my such make over our even most made after also did many before must through back "
    "years where much your way well down should because each just those people how too little "
    "state good very make world still own see men work long get here between both life being under "
    "never day same another know while last might us great old year off come since against go came "
    "right used take three river garden morning window letter village harbor lantern meadow orchard "
    "valley quiet bright narrow gentle distant silver amber hollow winter summer autumn spring stone "
    "bridge market candle shadow mirror pocket ribbon basket kettle chimney ladder blanket feather"
).split()


@dataclass(frozen=True)
class TaskConfig:
    depth: int = 10
    file_tokens: int = 1000
    n_decoy_chains: int = 2
    n_fake_clues: int = 2
    difficulty: str = "L1"
    key: str = "K7"
    t0_pad: int = 0            # extra filler tokens appended to the system prompt
    seed: int = 0

    def hash(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class TaskInstance:
    files: dict[str, str]                 # name -> true content
    silent_files: dict[str, str]          # chain name -> content with the valid clue redirected
    start: str
    answer: str
    chain: list[str]
    decoys: list[list[str]]
    decoy_answers: list[str]
    decoy_map: dict[str, str]             # chain name -> redirect target (file name or fake answer)
    key: str
    system_pad: str
    config: TaskConfig
    meta: dict = field(default_factory=dict)

    @property
    def depth(self) -> int:
        return len(self.chain)

    def index_of(self, name: str) -> int | None:
        try:
            return self.chain.index(name)
        except ValueError:
            return None


# ---------------------------------------------------------------------------

def _rand_str(rng: np.random.Generator, n: int, alphabet: str) -> str:
    return "".join(rng.choice(list(alphabet), size=n))


def filler_exact(rng: np.random.Generator, n_chars: int, line_len: int = 80) -> str:
    """Filler prose of exactly ``n_chars`` characters, broken into ~line_len lines."""
    if n_chars <= 0:
        return ""
    n_words = n_chars // 4 + 40                      # words average > 4 chars incl. separator
    idx = rng.integers(0, len(_WORDS), size=n_words)
    sent_len = rng.integers(6, 14, size=n_words // 6 + 2)
    words = [_WORDS[i] for i in idx]
    out, pos = [], 0
    for L in sent_len:
        chunk = words[pos:pos + L]
        if not chunk:
            break
        pos += L
        out.append(chunk[0].capitalize() + (" " + " ".join(chunk[1:]) if len(chunk) > 1 else "") + ".")
    text = " ".join(out)
    while len(text) < n_chars:                       # extremely rare top-up
        text += " " + text[: n_chars - len(text)]
    text = text[:n_chars]
    if text.endswith(" "):
        text = text[:-1] + "."
    chars = list(text)
    p = line_len
    while p < len(chars):
        q = text.find(" ", p)
        if q < 0:
            break
        chars[q] = "\n"
        p = q + line_len
    return "".join(chars)


def _clue_lines(kind: str, difficulty: str, key: str, target: str, rng) -> list[str]:
    """Render a clue (kind = 'next' | 'answer') as one or two lines."""
    if kind == "answer":
        return [f"[{key}] ANSWER: {target}"]
    if difficulty in ("L0", "L1"):
        return [f"[{key}] NEXT FILE: {target}"]
    rev = target[::-1]
    if difficulty == "L2":
        return [f"[{key}] NEXT FILE: the file whose name, reversed, is {rev}"]
    return [f"[{key}] NEXT FILE (part 1 of 2): the file whose name, reversed, begins with {rev[:3]}",
            f"[{key}] NEXT FILE (part 2 of 2): and, reversed, ends with {rev[3:]}"]


def _compose(rng, target_tokens: int, clue_groups: list[list[str]]) -> str:
    """Insert clue lines into filler so the file is exactly target_tokens tokens."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    clue_chars = sum(len(c) + 1 for g in clue_groups for c in g)
    filler = filler_exact(rng, max(target_chars - clue_chars, 0))
    lines = filler.split("\n") if filler else []
    for g in clue_groups:
        if len(g) == 1 or not lines:
            pos = int(rng.integers(0, len(lines) + 1))
            lines[pos:pos] = g
        else:  # split clue: part 1 then part 2 separated by >= 1 filler line
            p1 = int(rng.integers(0, max(1, len(lines) - 1)))
            p2 = int(rng.integers(p1 + 1, len(lines) + 1))
            lines.insert(p2, g[1])
            lines.insert(p1, g[0])
    return "\n".join(lines)


def generate(cfg: TaskConfig) -> TaskInstance:
    if cfg.difficulty not in DIFFICULTIES:
        raise ValueError(cfg.difficulty)
    if cfg.depth < 1:
        raise ValueError("depth must be >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([cfg.seed, int(cfg.hash(), 16) % (2**32)]))
    n = cfg.depth
    total = n * (1 + cfg.n_decoy_chains)
    names: set[str] = set()
    while len(names) < total:
        names.add(_rand_str(rng, 6, NAME_ALPHABET))
    names_l = sorted(names)          # sets of str iterate in hash order (PYTHONHASHSEED): sort first
    rng.shuffle(names_l)
    chain = names_l[:n]
    decoys = [names_l[n + j * n: n + (j + 1) * n] for j in range(cfg.n_decoy_chains)]
    answer = _rand_str(rng, 8, TOKEN_ALPHABET)
    decoy_answers = []
    while len(decoy_answers) < cfg.n_decoy_chains:
        a = _rand_str(rng, 8, TOKEN_ALPHABET)
        if a != answer and a not in decoy_answers:
            decoy_answers.append(a)
    key_digit = cfg.key[-1]
    fake_keys = [f"K{d}" for d in "0123456789" if d != key_digit]
    decoy_files = [x for dc in decoys for x in dc]
    use_fakes = cfg.difficulty != "L0" and cfg.n_fake_clues > 0

    def fake_groups() -> list[list[str]]:
        if not use_fakes or not decoy_files:
            return []
        out = []
        for _ in range(cfg.n_fake_clues):
            fk = str(rng.choice(fake_keys))
            tgt = str(rng.choice(decoy_files))
            out.append(_clue_lines("next", cfg.difficulty, fk, tgt, rng))
        return out

    files: dict[str, str] = {}
    silent: dict[str, str] = {}
    decoy_map: dict[str, str] = {}

    def build(name: str, kind: str, target: str, redirect: str | None):
        valid = _clue_lines(kind, cfg.difficulty, cfg.key, target, rng)
        fakes = fake_groups()
        file_seed = int(rng.integers(0, 2**32))
        files[name] = _compose(np.random.default_rng(file_seed), cfg.file_tokens, [valid] + fakes)
        if redirect is not None:
            bad = _clue_lines(kind, cfg.difficulty, cfg.key, redirect, rng)
            silent[name] = _compose(np.random.default_rng(file_seed), cfg.file_tokens, [bad] + fakes)

    for i, name in enumerate(chain):
        last = i == n - 1
        if cfg.n_decoy_chains:
            j = i % cfg.n_decoy_chains
            redirect = decoy_answers[j] if last else decoys[j][i + 1]
        else:
            redirect = ("ZZZZZZZZ" if last else "nofile")
        decoy_map[name] = redirect
        build(name, "answer" if last else "next", answer if last else chain[i + 1], redirect)
    for j, dc in enumerate(decoys):
        for i, name in enumerate(dc):
            last = i == n - 1
            build(name, "answer" if last else "next", decoy_answers[j] if last else dc[i + 1], None)

    pad = filler_exact(rng, cfg.t0_pad * CHARS_PER_TOKEN) if cfg.t0_pad else ""
    return TaskInstance(files=files, silent_files=silent, start=chain[0], answer=answer, chain=chain,
                        decoys=decoys, decoy_answers=decoy_answers, decoy_map=decoy_map, key=cfg.key,
                        system_pad=pad, config=cfg, meta={"config_hash": cfg.hash()})


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def system_prompt(task: TaskInstance) -> str:
    d = task.config.difficulty
    rules = [
        "You are navigating a directory of text files to find a secret answer token.",
        "Tools: list_files() lists file names; read_file(name) returns a file; answer(token) submits "
        "your final answer and ends the task.",
        f"Each file contains filler text plus clue lines. ONLY follow clue lines tagged [{task.key}]. "
        "Ignore clue lines with any other tag.",
        f"A [{task.key}] NEXT FILE clue tells you which file to read next. "
        f"When you read a line starting with '[{task.key}] ANSWER:', call answer with the exact "
        "token written after it. Do not call answer before you have read that line.",
    ]
    if d in ("L2", "L3"):
        rules.append("Some clues give the next file's name reversed; reverse it back before reading.")
    if d == "L3":
        rules.append("A clue may be split into two parts on separate lines; combine them.")
    rules.append("Read one file per step. Do not guess file names.")
    text = "\n".join(rules)
    if task.system_pad:
        text += "\n\nBackground notes (not relevant to the task):\n" + task.system_pad
    return text


def task_prompt(task: TaskInstance) -> str:
    return (f"Start by reading the file named {task.start}. Follow the chain of [{task.key}] clues "
            "until you find the answer token, then submit it with the answer tool.")


def single_shot_prompt(task: TaskInstance, rng: np.random.Generator | None = None) -> str:
    names = list(task.files)
    (rng or np.random.default_rng(0)).shuffle(names)
    parts = [f"=== {nm} ===\n{task.files[nm]}" for nm in names]
    return ("Below are all the files in the directory.\n\n" + "\n\n".join(parts) + "\n\n"
            f"Start at file {task.start} and follow the [{task.key}] clues to the answer. "
            "Reply with exactly one line: ANSWER: <token>")
