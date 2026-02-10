# src/commons/union_find.py
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1]*n
    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]
        return True
    def components(self):
        roots = {}
        for i in range(len(self.p)):
            r = self.find(i)
            roots.setdefault(r, []).append(i)
        return list(roots.values())
