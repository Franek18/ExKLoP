# If the parameter A's value is at least 20, the lowest possible value of parameter B is 5.
def r1(a: float, b: float) -> bool:
    if b / a >= 0.25:
        return True
    else:
        return False

# If the parameter C's value is at least 20, the lowest possible value of parameter D is 40.
def r2(c: float, d: float) -> bool:
    if d / c >= 2:
        return True
    else:
        return False

# Parameter E's value cannot be less than parameter F's value.
def r3(e: float, f: float) -> bool:
    if e >= f:
        return True
    else:
        return False

# Parameter G's value cannot be less than parameter H's value.
def r4(g: float, h: float) -> bool:
    if g >= h:
        return True
    else:
        return False

# Parameter I's value cannot be less than parameter J's value.
def r5(i: float, j: float) -> bool:
    if i >= j:
        return True
    else:
        return False

# Parameter K's value cannot be less than parameter L's value.
def r6(k: float, l: float) -> bool:
    if k >= l:
        return True
    else:
        return False

# Parameter M's value cannot be greater than parameter N's value.
def r7(m: float, n: float) -> bool:
    if m <= n:
        return True
    else:
        return False

# Parameter O's value cannot be greater than parameter P's value.
def r8(o: float, p: float) -> bool:
    if o <= p:
        return True
    else:
        return False

# Parameter R's value cannot be greater than parameter S's value.
def r9(r: float, s: float) -> bool:
    if r <= s:
        return True
    else:
        return False

# Parameter T's value cannot be greater than parameter U's value.
def r10(t: float, u: float) -> bool:
    if t <= u:
        return True
    else:
        return False

# Parameter W cannot be less than the sum of parameter X, parameter Y and parameter Z.
def r11(w: float, x: float, y: float, z: float) -> bool:
    if w >= x + y + z:
        return True
    else:
        return False