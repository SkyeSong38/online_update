import os
import sys
path = "/home/szk/Developer/trans_tracking/CSTrT"
print("---"+path)
sys.path.append(path)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from lib.test.vot20.cstrt_vot20 import run_vot_exp
run_vot_exp()