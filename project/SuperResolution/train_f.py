# -*- coding: utf-8 -*-

import caffe
import time

start_time = time.time()
solver_path = 'examples/FSRCNN/FSRCNN_solver.prototxt'
solver = caffe.SGDSolver(solver_path);
solver.solve();

print("--- %s seconds ---" % (time.time() - start_time))