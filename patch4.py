with open('app/services/zmq_server.py', 'r') as f:
    zp = f.read()

zmq_old = 'm_cfg.T_PYRAMID = [t, t*2]'
zmq_new = 'm_cfg.T_PYRAMID = [t, t*2, t*4, t*8, t*16]\n                m_cfg.PYRAMID_LEVELS = 5'

zp = zp.replace(zmq_old, zmq_new)
with open('app/services/zmq_server.py', 'w') as f:
    f.write(zp)
print('ZMQ patched')
