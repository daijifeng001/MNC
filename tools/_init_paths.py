
import os.path
import sys

"""
Add lib paths and caffe path to system search path
"""


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

cur_dir = os.path.dirname(__file__)

# Add caffe python to PYTHONPATH
caffe_path = os.path.join(cur_dir, '..', 'caffe-mnc', 'python')
add_path(caffe_path)

# Add lib to PYTHONPATH
lib_path = os.path.join(cur_dir, '..', 'lib')
add_path(lib_path)
