import sys, numpy as np
cases=[("GDN qkv",10240),("GDN out",2560),("GDN z",6144),("QSA out",2560),("QSA q",12288),("shexp dn",2560),("shexp gate",640),("GDN qkv5",10240),("shexp q6",640),("shexp xs",640)]
a=np.fromfile(sys.argv[1],np.float32); b=np.fromfile(sys.argv[2],np.float32); o=0
for name,n in cases:
    cols={}
    for nc in (1,2,3,4):
        sz=n*nc; A=a[o:o+sz].reshape(nc,n); Bv=b[o:o+sz].reshape(nc,n); o+=sz; cols[nc]=Bv
        rel=np.abs(A-Bv).max()/np.abs(A).max()
        if nc==4: print(f"{name:9s} max rel diff vs old {rel:.2e}", end="")
    exact=all(np.array_equal(cols[nc][j],cols[4][j]) for nc in (1,2,3) for j in range(nc))
    print(f"   columns exact across T: {exact}")
