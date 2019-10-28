def prepare_delayscreen(delta, data, prop, logstr):
    #Assign variables
    nx = prop.nx;
    ny = prop.ny;
    dx = prop.dx;
    dy = prop.dy;
    dt = prop.dt;
    abflag = prop.abflag;
    c0 = prop.mat.c0;
    step = prop.stepsize;

    raise NotImplementedError