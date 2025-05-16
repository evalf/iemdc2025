import functools
import itertools
import numpy
from nutils import evaluable, matrix
import treelog

def _ravel(f, ndim):
    if ndim == 0:
        return evaluable.InsertAxis(f, evaluable.constant(1))
    for i in range(ndim - 1):
        f = evaluable.Ravel(f)
    return f

def _as_block_csr(blocks):
    if isinstance(blocks, (tuple, list)):
        return tuple(map(_as_block_csr, blocks))
    else:
        return evaluable.as_csr(blocks.simplified)

def _simplified(blocks):
    if isinstance(blocks, (tuple, list)):
        return tuple(map(_simplified, blocks))
    else:
        return blocks.simplified

def _is_invariant_to(blocks, arguments):
    if isinstance(blocks, (tuple, list)):
        return all(_is_invariant_to(block, arguments) for block in blocks)
    else:
        return blocks.arguments.isdisjoint(arguments)

def _assemble(blocks, dtype, *masks):
    assert masks
    assert all(mask.ndim == 1 and mask.dtype == bool for mask in masks)
    if any(mask.size == 0 for mask in masks):
        return numpy.zeros(tuple(numpy.sum(mask) for mask in masks), dtype)
    return _assemble_helper(blocks, 0, *masks)

def _assemble_helper(blocks, axis, mask, *inner_masks):
    assert isinstance(blocks, (tuple, list))
    if inner_masks:
        blocks = [_assemble_helper(inner_blocks, axis+1, *inner_masks) for inner_blocks in blocks]
    return numpy.concatenate(blocks, axis=axis)[(slice(None),)*axis + (mask,)]

@treelog.withcontext
def minimize(scalar, residual, target, trial, test, *, tol, constrain=None, arguments=None):
    scalar = scalar.as_evaluable_array
    residual = residual.as_evaluable_array
    arg_objects = {
        arg.name: arg
        for fun in (scalar, residual)
        for arg in fun.arguments
        if isinstance(arg, evaluable.Argument)
    }

    assert scalar.ndim == 0
    assert residual.ndim == 0

    if isinstance(target, str):
        target = target.split(',')
    if isinstance(trial, str):
        trial = trial.split(',')
    if isinstance(test, str):
        test = test.split(',')

    x = []
    y = []
    for name in target:
        arg = arg_objects[name]
        (y if arg in residual.arguments else x).append(arg)
    z = [arg_objects[name] for name in trial]

    dtype, = {arg.dtype for arg in itertools.chain(x, y, z)}

    def d(func, *targets):
        if len(targets) == 0:
            return func
        if isinstance(func, (tuple, list)):
            return [d(func_i, *targets) for func_i in func]
        if len(targets) > 1:
            return d(d(func, *targets[:-1]), targets[-1])
        return [_ravel(evaluable.derivative(func, t), t.ndim) for t in targets[0]]

    arguments = dict(arguments) if arguments else {}
    free = {}
    for arg in itertools.chain(x, y, z):
        shape = evaluable.eval_once(arg.shape)
        val = numpy.zeros(shape, dtype)
        if (init_val := arguments.get(arg.name)) is not None:
            val[...] = init_val
        arguments[arg.name] = val
        if constrain and (c := constrain.get(arg.name)) is not None:
            assert c.shape == shape
            free[arg.name] = m = numpy.isnan(c)
            val[~m] = c[~m]
        else:
            free[arg.name] = numpy.ones(shape, dtype=bool)

    x_free, y_free, z_free = (
        numpy.concatenate([free[arg.name].ravel() for arg in args]) if args else numpy.zeros((0,), dtype=bool)
        for args in (x, y, z)
    )
    nx_free, ny_free, nz_free = map(numpy.sum, (x_free, y_free, z_free))

    f = scalar
    g = d(residual, [arg_objects[name] for name in test])

    all_funcs = [
        f,
        d(f,x), d(f,y), d(f,z),
        d(f,x,x), d(f,x,y), d(f,x,z), d(f,y,y), d(f,y,z), _as_block_csr(_simplified(d(f,z,z))),
        g, d(g,y), _as_block_csr(_simplified(d(g,z))),
    ]
    all_funcs = _simplified(all_funcs)
    min_args = set(itertools.chain(x, y, z))
    is_variable = [not _is_invariant_to(func, min_args) for func in all_funcs]
    eval_all = evaluable.compile(all_funcs)
    eval_var = evaluable.compile([func for func, iv in zip(all_funcs, is_variable) if iv])

    info_parts = []
    if x:
        info_parts.append('for {}'.format(' and '.join(f'{arg.name} {free[arg.name].shape}' for arg in x)))
    if y:
       info_parts.append('for {} via {}'.format(' and '.join(f'{arg.name} {free[arg.name].shape}' for arg in y), ' and '.join(f'{arg.name} {free[arg.name].shape}' for arg in z)))
    treelog.info('minimizing {}'.format(' and '.join(info_parts)))

    dyz = numpy.zeros((nz_free, ny_free), dtype)
    update_dyz = True

    for i in itertools.count():
        with treelog.context(f'iter {i}'):
            if i > 20:
                raise Exception('maximum number of iterations reached')
            if i == 0:
                values = iter(eval_all(arguments))
                is_evaluated = itertools.repeat(True)
            else:
                values = iter(eval_var(arguments))
                is_evaluated = iter(is_variable)
            if next(is_evaluated):
                f = next(values)
            if next(is_evaluated):
                dxf = _assemble(next(values), dtype, x_free)
            if next(is_evaluated):
                dyf = _assemble(next(values), dtype, y_free)
            if next(is_evaluated):
                dzf = _assemble(next(values), dtype, z_free)
            if next(is_evaluated):
                dxxf = _assemble(next(values), dtype, x_free, x_free)
            if next(is_evaluated):
                dxyf = _assemble(next(values), dtype, x_free, y_free)
            if next(is_evaluated):
                dxzf = _assemble(next(values), dtype, x_free, z_free)
            if next(is_evaluated):
                dyyf = _assemble(next(values), dtype, y_free, y_free)
            if next(is_evaluated):
                dyzf = _assemble(next(values), dtype, y_free, z_free)
            if next(is_evaluated):
                dzzf = matrix.assemble_block_csr(next(values)).submatrix(z_free, z_free)
            if next(is_evaluated):
                g = _assemble(next(values), dtype, z_free)
            if next(is_evaluated):
                dyg = _assemble(next(values), dtype, z_free, y_free)
                update_dyz = True
            if next(is_evaluated):
                dzg = matrix.assemble_block_csr(next(values)).submatrix(z_free, z_free)
                update_dyz = True

            if update_dyz:
                for j in range(dyz.shape[1]):
                    dyz[:,j] = -dzg.solve(dyg[:,j])
                update_dyz = False

            grad = numpy.concatenate([dxf, dyf + dzf @ dyz, g])
            norm_grad = numpy.linalg.norm(grad)

            treelog.info(f"scalar = {f:.3e}, norm([grad(scalar), residual]) = {norm_grad:.3e}")
            if norm_grad <= tol:
                break

            # [A   B   C] [Δx]   [u]
            # [D   E   F] [Δy] = [v]
            # [    G   H] [Δz]   [w]
            #
            # Substituting Δz = H_inv (w - G Δy):
            #
            # [A   B - C H_inv G] [Δx]   [u - C H_inv w]
            # [D   E - F H_inv G] [Δy] = [v - F H_inv w]

            A = dxxf
            B = dxyf
            C = dxzf
            D = dxyf.T + (dxzf @ dyz).T
            E = dyyf + (dyzf @ dyz).T
            F = dyzf + (dzzf @ dyz).T
            G = dyg
            H = dzg
            u = -dxf
            v = -dyf - dzf @ dyz
            w = -g

            H_inv_G = -dyz
            H_inv_w = H.solve(w)

            J = numpy.block([[A, B - C @ H_inv_G], [D, E - F @ H_inv_G]])
            r = numpy.concatenate([u - C @ H_inv_w, v - F @ H_inv_w])
            Δxy = numpy.linalg.solve(J, r)
            Δx = Δxy[:nx_free]
            Δy = Δxy[nx_free:]
            Δz = H.solve(w - G @ Δy)

            for args, Δval in (x, Δx), (y, Δy), (z, Δz):
                start = 0
                for arg in args:
                    stop = start + numpy.sum(free[arg.name])
                    numpy.add.at(arguments[arg.name], free[arg.name], Δval[start:stop])
                    start = stop
                assert start == Δval.size

    return arguments

if __name__ == '__main__':
    import numpy
    from nutils.function import Argument
    x = Argument('x', ())
    y = Argument('y', ())
    z = Argument('z', ())
    f = numpy.sum((y - 1)**2)
    g = numpy.sum(z * (y - x - 2))
    minimize(f, g, 'x', 'y', 'z', tol=1e-4)

    from nutils import mesh
    from nutils.expression_v2 import Namespace
    from nutils.solver import System
    ns = Namespace()
    X, ns.x = mesh.rectilinear([4])
    ns.define_for('x', gradient='∇', jacobians=('dV', 'dS'), normal='n')
    ns.w = Argument('w', ())
    ns.u = X.field('u', btype='spline', degree=2)
    ns.v = X.field('v', btype='spline', degree=2)
    R = X.integral('(∇_i(v) ∇_i(u) - v w) dV' @ ns, degree=2)
    f = (X.integral('u dV' @ ns, degree=2) - 1)**2
    cons = System(X.boundary.integral('u^2 dS' @ ns, degree=4), 'u').solve_constraints(droptol=1e-10)
    args = minimize(f, R, 'w', 'u', 'v', tol=1e-4, constrain=cons)
