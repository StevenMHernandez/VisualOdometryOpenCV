from graphslam.load import load_g2o_se3

g = load_g2o_se3("../output/output.g2o")

g.plot(vertex_markersize=1)
g.calc_chi2()
g.optimize(max_iter=1)
g.plot(vertex_markersize=1)
g.optimize(max_iter=1)
g.plot(vertex_markersize=1)
