from scha import SCHAParams, build_scha_state_space

p = SCHAParams(m=2, n=4, lambdas=[0.5])
ss = build_scha_state_space(p)
print('state dim:', ss['dim'])
