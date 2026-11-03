"""Small, self-contained Python function-writing tasks with hidden unit tests (E10).

Each task: a spec shown to the agent, the function name, hidden test cases
``(args, expected)``, and a reference solution (used only by the mock client and
by the task self-test, never shown to the agent).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodingTask:
    id: str
    func: str
    spec: str
    tests: tuple
    reference: str


def _t(id, func, spec, tests, reference):
    return CodingTask(id, func, spec.strip(), tuple(tests), reference.strip() + "\n")


TASKS: list[CodingTask] = [
    _t("rev_words", "reverse_words",
       "Write `reverse_words(s: str) -> str` returning the words of s in reverse order, separated by single "
       "spaces. Leading/trailing/multiple spaces are ignored.",
       [(("hello world",), "world hello"), (("  a  b c ",), "c b a"), (("",), ""), (("one",), "one")],
       "def reverse_words(s):\n    return ' '.join(s.split()[::-1])"),
    _t("fizz", "fizzbuzz",
       "Write `fizzbuzz(n: int) -> list[str]` for 1..n: 'Fizz' for multiples of 3, 'Buzz' for 5, "
       "'FizzBuzz' for both, else the number as a string.",
       [((5,), ["1", "2", "Fizz", "4", "Buzz"]), ((0,), []), ((15,), ["1", "2", "Fizz", "4", "Buzz", "Fizz", "7",
         "8", "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz"])],
       "def fizzbuzz(n):\n    out=[]\n    for i in range(1,n+1):\n        s=('Fizz' if i%3==0 else '')+('Buzz' if i%5==0 else '')\n        out.append(s or str(i))\n    return out"),
    _t("palin", "is_palindrome",
       "Write `is_palindrome(s: str) -> bool`: True if s reads the same backwards considering only "
       "alphanumeric characters, case-insensitively.",
       [(("A man, a plan, a canal: Panama",), True), (("race a car",), False), (("",), True), (("ab_a",), True)],
       "def is_palindrome(s):\n    t=[c.lower() for c in s if c.isalnum()]\n    return t==t[::-1]"),
    _t("anagram", "group_anagrams",
       "Write `group_anagrams(words: list[str]) -> list[list[str]]` grouping anagrams. Each group keeps the "
       "input order; groups are ordered by the first appearance of their first word.",
       [((["eat", "tea", "tan", "ate", "nat", "bat"],), [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]),
        (([],), []), ((["a"],), [["a"]])],
       "def group_anagrams(words):\n    g={}\n    for w in words:\n        g.setdefault(''.join(sorted(w)),[]).append(w)\n    return list(g.values())"),
    _t("rle", "run_length_encode",
       "Write `run_length_encode(s: str) -> str` producing e.g. 'aaabcc' -> 'a3b1c2'.",
       [(("aaabcc",), "a3b1c2"), (("",), ""), (("z",), "z1"), (("aabbaa",), "a2b2a2")],
       "def run_length_encode(s):\n    out=[]\n    i=0\n    while i<len(s):\n        j=i\n        while j<len(s) and s[j]==s[i]:\n            j+=1\n        out.append(f'{s[i]}{j-i}')\n        i=j\n    return ''.join(out)"),
    _t("primes", "primes_upto",
       "Write `primes_upto(n: int) -> list[int]` returning all primes <= n in increasing order.",
       [((10,), [2, 3, 5, 7]), ((1,), []), ((2,), [2]), ((30,), [2, 3, 5, 7, 11, 13, 17, 19, 23, 29])],
       "def primes_upto(n):\n    if n<2: return []\n    s=[True]*(n+1); s[0]=s[1]=False\n    for i in range(2,int(n**0.5)+1):\n        if s[i]:\n            for j in range(i*i,n+1,i): s[j]=False\n    return [i for i,v in enumerate(s) if v]"),
    _t("flatten", "flatten",
       "Write `flatten(x: list) -> list` that flattens arbitrarily nested lists (only lists are nested).",
       [(([1, [2, [3, [4]], 5]],), [1, 2, 3, 4, 5]), (([],), []), (([[[]]],), [])],
       "def flatten(x):\n    out=[]\n    for v in x:\n        out.extend(flatten(v) if isinstance(v,list) else [v])\n    return out"),
    _t("roman", "to_roman",
       "Write `to_roman(n: int) -> str` converting 1 <= n <= 3999 to a Roman numeral.",
       [((1,), "I"), ((4,), "IV"), ((1994,), "MCMXCIV"), ((3999,), "MMMCMXCIX")],
       "def to_roman(n):\n    v=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out=''\n    for a,s in v:\n        while n>=a:\n            out+=s; n-=a\n    return out"),
    _t("from_roman", "from_roman",
       "Write `from_roman(s: str) -> int` converting a valid Roman numeral to an integer.",
       [(("III",), 3), (("IV",), 4), (("MCMXCIV",), 1994), (("LVIII",), 58)],
       "def from_roman(s):\n    v={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}\n    t=0\n    for i,c in enumerate(s):\n        if i+1<len(s) and v[c]<v[s[i+1]]: t-=v[c]\n        else: t+=v[c]\n    return t"),
    _t("brackets", "balanced",
       "Write `balanced(s: str) -> bool`: True iff the brackets ()[]{} in s are balanced and properly nested; "
       "other characters are ignored.",
       [(("([]{})",), True), (("([)]",), False), (("",), True), (("a(b)c]",), False)],
       "def balanced(s):\n    m={')':'(',']':'[','}':'{'}\n    st=[]\n    for c in s:\n        if c in '([{': st.append(c)\n        elif c in m:\n            if not st or st.pop()!=m[c]: return False\n    return not st"),
    _t("merge_iv", "merge_intervals",
       "Write `merge_intervals(iv: list[list[int]]) -> list[list[int]]` merging overlapping closed intervals; "
       "return them sorted by start.",
       [(([[1, 3], [2, 6], [8, 10], [15, 18]],), [[1, 6], [8, 10], [15, 18]]), (([[1, 4], [4, 5]],), [[1, 5]]),
        (([],), [])],
       "def merge_intervals(iv):\n    out=[]\n    for a,b in sorted(iv):\n        if out and a<=out[-1][1]: out[-1][1]=max(out[-1][1],b)\n        else: out.append([a,b])\n    return out"),
    _t("two_sum", "two_sum",
       "Write `two_sum(nums: list[int], target: int) -> list[int]` returning indices [i, j], i < j, of the first "
       "pair (by j, then i) summing to target, or [] if none.",
       [(([2, 7, 11, 15], 9), [0, 1]), (([3, 2, 4], 6), [1, 2]), (([1, 2], 7), [])],
       "def two_sum(nums,target):\n    seen={}\n    for j,x in enumerate(nums):\n        if target-x in seen: return [seen[target-x],j]\n        seen.setdefault(x,j)\n    return []"),
    _t("caesar", "caesar",
       "Write `caesar(s: str, k: int) -> str` shifting letters by k (wrapping, preserving case); other "
       "characters unchanged. k may be negative.",
       [(("abc", 1), "bcd"), (("XyZ", 2), "ZaB"), (("Hello, World!", -1), "Gdkkn, Vnqkc!")],
       "def caesar(s,k):\n    out=[]\n    for c in s:\n        if c.isalpha():\n            b=ord('A') if c.isupper() else ord('a')\n            out.append(chr((ord(c)-b+k)%26+b))\n        else: out.append(c)\n    return ''.join(out)"),
    _t("word_freq", "top_words",
       "Write `top_words(text: str, k: int) -> list[str]` returning the k most frequent lowercase words "
       "(split on non-letters); ties broken alphabetically.",
       [(("b a b c a b", 2), ["b", "a"]), (("The the THE cat", 1), ["the"]), (("", 3), [])],
       "import re\nfrom collections import Counter\ndef top_words(text,k):\n    c=Counter(w.lower() for w in re.findall(r'[A-Za-z]+',text))\n    return [w for w,_ in sorted(c.items(), key=lambda x:(-x[1],x[0]))[:k]]"),
    _t("matrix_t", "transpose",
       "Write `transpose(m: list[list[int]]) -> list[list[int]]` for a rectangular matrix.",
       [(([[1, 2, 3], [4, 5, 6]],), [[1, 4], [2, 5], [3, 6]]), (([],), []), (([[1]],), [[1]])],
       "def transpose(m):\n    return [list(r) for r in zip(*m)]"),
    _t("binsearch", "bisect_left",
       "Write `bisect_left(a: list[int], x: int) -> int` returning the leftmost insertion index of x in sorted a.",
       [(([1, 2, 2, 3], 2), 1), (([], 5), 0), (([1, 3], 4), 2), (([1, 3], 0), 0)],
       "def bisect_left(a,x):\n    lo,hi=0,len(a)\n    while lo<hi:\n        m=(lo+hi)//2\n        if a[m]<x: lo=m+1\n        else: hi=m\n    return lo"),
    _t("digits", "digit_root",
       "Write `digit_root(n: int) -> int` repeatedly summing the decimal digits of n >= 0 until one digit remains.",
       [((0,), 0), ((38,), 2), ((999999,), 9), ((10,), 1)],
       "def digit_root(n):\n    while n>=10: n=sum(map(int,str(n)))\n    return n"),
    _t("chunks", "chunk",
       "Write `chunk(xs: list, n: int) -> list[list]` splitting xs into consecutive chunks of size n (last may be "
       "shorter).",
       [(([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]]), (([], 3), []), (([1, 2], 5), [[1, 2]])],
       "def chunk(xs,n):\n    return [xs[i:i+n] for i in range(0,len(xs),n)]"),
    _t("lcp", "longest_common_prefix",
       "Write `longest_common_prefix(words: list[str]) -> str` (empty string for an empty list).",
       [((["flower", "flow", "flight"],), "fl"), ((["dog", "car"],), ""), (([],), ""), ((["a"],), "a")],
       "def longest_common_prefix(words):\n    if not words: return ''\n    p=words[0]\n    for w in words[1:]:\n        while not w.startswith(p): p=p[:-1]\n    return p"),
    _t("valid_ip", "is_ipv4",
       "Write `is_ipv4(s: str) -> bool`: four decimal octets 0-255 separated by dots, no leading zeros "
       "(except '0' itself), digits only.",
       [(("192.168.0.1",), True), (("256.1.1.1",), False), (("01.2.3.4",), False), (("1.2.3",), False),
        (("0.0.0.0",), True)],
       "def is_ipv4(s):\n    p=s.split('.')\n    if len(p)!=4: return False\n    for q in p:\n        if not q.isdigit() or (len(q)>1 and q[0]=='0') or int(q)>255: return False\n    return True"),
    _t("spiral", "spiral_order",
       "Write `spiral_order(m: list[list[int]]) -> list[int]` returning elements in clockwise spiral order.",
       [(([[1, 2, 3], [4, 5, 6], [7, 8, 9]],), [1, 2, 3, 6, 9, 8, 7, 4, 5]), (([],), []),
        (([[1, 2], [3, 4], [5, 6]],), [1, 2, 4, 6, 5, 3])],
       "def spiral_order(m):\n    out=[]\n    m=[r[:] for r in m]\n    while m:\n        out+=m.pop(0)\n        m=[list(r) for r in zip(*m)][::-1]\n    return out"),
    _t("camel", "snake_to_camel",
       "Write `snake_to_camel(s: str) -> str`: 'hello_big_world' -> 'helloBigWorld'. Ignore empty segments.",
       [(("hello_big_world",), "helloBigWorld"), (("a",), "a"), (("__x__y",), "xY"), (("",), "")],
       "def snake_to_camel(s):\n    p=[x for x in s.split('_') if x]\n    return (p[0]+''.join(w.capitalize() for w in p[1:])) if p else ''"),
]
