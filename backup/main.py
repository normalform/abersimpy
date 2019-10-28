from backup.plan_simulation import plan_and_run_simulation


if __name__ == '__main__':
    fnstr = '{}.mat'.format('tutorial2')
    logstr = '{}.log'.format('trace')

    plan_and_run_simulation(fnstr, logstr)