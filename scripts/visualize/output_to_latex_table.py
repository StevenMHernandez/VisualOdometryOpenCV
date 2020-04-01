import pandas as pd

X = pd.read_csv("../../output/output.csv", index_col="change")


selections = [
    [0,3,0],
    [0,6,0],
    [0,9,0],
    [0,0,3],
    [0,0,6],
    [0,0,9],
]

for i in selections:
    s = X.loc["x y z: (0 0 0) φ θ ψ: ({} {} {})".format(i[0], i[1], i[2])]
    out = "({},{},{}) & ({:.3f},{:.3f}) & ({:.3f},{:.3f}) & ({:.3f},{:.3f}) \\\\\\hline".format(
        i[0], i[1], i[2],
        s["φ.mean"], s["φ.std"],
        s["θ.mean"], s["θ.std"],
        s["ψ.mean"], s["ψ.std"],
    )
    print(out)


print("\n\n\n\n")

selections = [
    [0,100,0],
    [0,200,0],
    [0,300,0],
]

for i in selections:
    s = X.loc["x y z: ({} {} {}) φ θ ψ: (0 0 0)".format(i[0], i[1], i[2])]
    out = "({},{},{}) & ({:.3f},{:.3f}) & ({:.3f},{:.3f}) & ({:.3f},{:.3f}) \\\\\\hline".format(
        i[0], i[1], i[2],
        s[" X.mean"], s["X.std"],
        s["Y.mean"], s["Y.std"],
        s["Z.mean"], s["Z.std"],
    )
    print(out)


# (0,3,0) &&&\\\hline