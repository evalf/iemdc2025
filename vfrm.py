import dataclasses
from dataclasses import dataclass
from functools import cached_property, reduce
import math
import matplotlib.colors
import matplotlib.patches
from matplotlib.collections import LineCollection
import numpy
import nutils
from nutils import export, function, mesh, SI, sample
from nutils.expression_v2 import Namespace
from nutils.element import getsimplex
from nutils.pointsseq import PointsSequence
from nutils.SI import Area, CurrentDensity, Dimension, ElectricCurrent, ElectricPotential, Force, Length, MagneticFieldStrength, MagneticFluxDensity, Resistance, Time, Quantity, Power
from nutils.sample import Sample
from nutils.solver import System, LinesearchNewton
from nutils.transformseq import PlainTransforms
import operator
from pathlib import Path
import scipy.optimize
import treelog
from types import SimpleNamespace
from typing import Union

Angle = Dimension.create('Φ')
SI.units.rad = Angle.wrap(1.)
SI.units.deg = numpy.pi / 180 * SI.units.rad

AngularFrequency = Angle / Time
SI.units.rpm = AngularFrequency('360deg/min')

Permeability = MagneticFluxDensity / MagneticFieldStrength
Torque = Force * Length
Resistivity = Resistance * Length
Conductivity = Resistivity**-1

mu0 = 4e-7 * numpy.pi * Permeability('H/m')

plt_length = 'mm'
plt_mag_pot = 'T*mm'
plt_flux = 'mWb'
plt_flux_dens = 'T'
plt_elec_pot = 'V'
plt_current = 'A'
plt_cur_dens = 'A/mm2'
plt_angle = 'deg'
plt_torque = 'N*m'


@dataclass(frozen=True)
class Geometry:
    radius_rotor_inner: Length = Length('10.5mm')
    height_rotor_yoke: Length = Length('20.5mm')
    height_rotor_teeth: Length = Length('14.4mm')
    height_air_gap: Length = Length('0.535mm')
    height_ac_edge: Length = Length('7.75mm')
    height_dc_edge: Length = Length('7.75mm')
    height_ac_center: Length = Length('9mm')
    height_dc_center: Length = Length('7.75mm')
    height_stator: Length = Length('24.1mm')
    arc_length_stator_teeth_inner: Length = Length('13mm')
    arc_length_rotor_teeth_inner: Length = Length('5mm')
    arc_length_rotor_teeth_outer: Length = Length('10mm')
    n_stator_teeth: int = 12
    n_rotor_teeth: int = 10
    element_size: Length = Length('10mm')
    n_elements_air_gap: int = 1
    stack_length: Length = Length('87.5mm')
    interface_position: float = 0.5
    n_turns_ac: int = 10
    n_turns_dc: int = 14
    wire_area: Area = Area('3.3mm2')
    wire_conductivity: Conductivity = Conductivity('59.6MS/m')

    @property
    def radius_stator_outer(self):
        return self.radius_rotor_inner + self.height_rotor_yoke + self.height_rotor_teeth + self.height_air_gap + self.height_stator

    @property
    def structural_multiplicity(self):
        return math.gcd(self.n_stator_teeth, self.n_rotor_teeth)

    @property
    def n_stator_teeth_div_mult(self):
        return self.n_stator_teeth // self.structural_multiplicity

    @property
    def n_rotor_teeth_div_mult(self):
        return self.n_rotor_teeth // self.structural_multiplicity

    @property
    def ac_conductance(self):
        ac_wire_length = (self.n_stator_teeth // 3) * self.n_turns_ac * 2 * self.stack_length
        return self.wire_conductivity * self.wire_area / ac_wire_length

    @property
    def ac_resistance(self):
        return 1 / self.ac_conductance

    @property
    def dc_conductance(self):
        dc_wire_length = self.n_stator_teeth * self.n_turns_dc * 2 * self.stack_length
        return self.wire_conductivity * self.wire_area / dc_wire_length

    @property
    def dc_resistance(self):
        return 1 / self.dc_conductance

    @treelog.withcontext
    def mesh(self, space: str):
        ref_length = Length('m')
        numbers = dict(
            r_ri=self.radius_rotor_inner / ref_length,
            h_ry=self.height_rotor_yoke / ref_length,
            h_rt=self.height_rotor_teeth / ref_length,
            h_a=self.height_air_gap / ref_length,
            h_sea=self.height_ac_edge / ref_length,
            h_sed=self.height_dc_edge / ref_length,
            h_sca=self.height_ac_center / ref_length,
            h_scd=self.height_dc_center / ref_length,
            h_s=self.height_stator / ref_length,
            a_rti=self.arc_length_rotor_teeth_inner / ref_length,
            a_rto=self.arc_length_rotor_teeth_outer / ref_length,
            n_rt=self.n_rotor_teeth_div_mult,
            a_sti=self.arc_length_stator_teeth_inner / ref_length,
            n_st=self.n_stator_teeth_div_mult,
            h_e=self.element_size / ref_length,
            n_ea=self.n_elements_air_gap,
            m=self.structural_multiplicity,
            f_i=self.interface_position,
        )
        topo, geom = mesh.gmsh(
            Path(__file__).parent/'vfrm.geo',
            dimension=2,
            order=2,
            space=space,
            numbers=numbers,
        )
        geom *= ref_length

        # Define subdomains for coils: `ac_{a,b,c}_{r,l}` for the ac coils of
        # phases `a`, `b` and `c` and `dc_{r,l}` for the dc coils.
        topo = topo \
            .withsubdomain(iron='stator_iron,rotor_iron') \
            .withsubdomain(**{f'ac_{chr(ord("a")+i)}_{s}': ','.join(f'ac_{j}_{s}' for j in range(i, self.n_stator_teeth_div_mult, 3)) for i in range(3) for s in 'pn'}) \
            .withsubdomain(**{f'dc_{s}': ','.join(f'dc_{j}_{s}' for j in range(self.n_stator_teeth_div_mult)) for s in 'pn'}) \
            .withsubdomain(**{f'{k}': f'{k}_p,{k}_n' for k in ('ac_a', 'ac_b', 'ac_c', 'dc')}) \
            .withsubdomain(**{f'{s}': ','.join(f'{k}_{s}' for k in ('ac_a', 'ac_b', 'ac_c', 'dc')) for s in 'pn'})

        return topo, geom


@dataclass(frozen=True)
class Driver:
    peak_ac_current: ElectricCurrent = ElectricCurrent('28.868A')
    dc_current: ElectricCurrent = ElectricCurrent('50A')
    commutation_angle: Angle = Angle('90deg')
    mechanical_angular_frequency: AngularFrequency = AngularFrequency('1000rpm')

    def ac_current(self, geometry: Geometry, mechanical_angle):
        electrical_angle = geometry.n_rotor_teeth * mechanical_angle - self.commutation_angle + numpy.stack([0, -120, 120]) * Angle('deg')
        return self.peak_ac_current * cos(electrical_angle)


class Machine:

    def __init__(
        self,
        geometry: Geometry = Geometry(),
        driver: Driver = Driver(),
        nelems_angle: int = 6,
        degree: int = 2,
    ):

        # REFERENCE LENGTHS

        ref_length = geometry.radius_stator_outer
        ref_area = numpy.pi * geometry.radius_stator_outer**2
        ref_surf = 2 * numpy.pi * geometry.radius_stator_outer
        ref_flux_dens = MagneticFluxDensity('T')
        ref_mag_pot = ref_flux_dens * ref_length
        ref_field_strength = ref_flux_dens / mu0
        ref_elec_pot = ElectricPotential('V')

        # ROTATION MESH

        T, thetam = mesh.line(nelems_angle, space=f'T', bnames=['begin', 'end'])
        nthetamperiods = geometry.n_stator_teeth_div_mult * geometry.n_rotor_teeth_div_mult
        thetamperiod = Angle('360deg') / geometry.structural_multiplicity / nthetamperiods
        thetam *= thetamperiod / T.shape[0]

        # SPACE MESH

        X, x0 = geometry.mesh(space='X')
        xrot = numpy.stack([cos(thetam) * x0[0] - sin(thetam) * x0[1], sin(thetam) * x0[0] + cos(thetam) * x0[1]])
        x = numpy.choose(X.indicator('rotating'), [x0 / ref_length, xrot / ref_length]) * ref_length

        # MAGNETIC RELUCTIVITY OF IRON

        nuiron = lambda Bmagsqr: 1 / (5000 * mu0)
        #nuiron = lambda Bmagsqr: numpy.exp(4.41797133 + 1.16730888 * Bmagsqr / 'T2') / Permeability('T*m/A')

        #Hiron, Biron = numpy.loadtxt('BHcurve_NO27.txt').T
        #Hiron *= MagneticFieldStrength('A/m')
        #Biron *= MagneticFluxDensity('T')
        #nuiron = lambda Bmagsqr: numpy.interp(Bmagsqr, Biron[1:]**2, Hiron[1:] / Biron[1:])

        # NAMESPACE

        ns = Namespace()
        ns.cos = cos
        ns.sin = sin
        ns.δ = function.eye(2)
        ns.D = function.levicivita(2)
        ns.mu0 = mu0
        ns.omegam = driver.mechanical_angular_frequency
        ns.nturnsac = geometry.n_turns_ac
        ns.nturnsdc = geometry.n_turns_dc
        ns.rad = Angle('rad')

        ns.δp = X.indicator('p')
        ns.δn = X.indicator('n')
        ns.δdc = X.indicator('dc')
        ns.δac = numpy.stack([X.indicator(f'ac_{k}') for k in 'abc'])

        ns.thetam = thetam
        ns.t = thetam / driver.mechanical_angular_frequency
        ns.x0 = x0
        ns.x = x
        ns.r = numpy.linalg.norm(ns.x)
        ns.phi = arctan2(ns.x[1], ns.x[0])
        ns.define_for('thetam', gradient='dthetam', jacobians=('dThetam',))
        ns.define_for('t', gradient='dt', jacobians=('dT',))
        ns.define_for('x0', gradient='grad0', jacobians=('dV0', 'dS0'), normal='n0', spaces=X.spaces)
        ns.define_for('x', gradient='grad', jacobians=('dV', 'dS'), normal='n', spaces=X.spaces)
        ns.nr_i = 'x_i / r'
        ns.nphi_i = 'D_ji nr_j'

        ns.A2 = function.field('A2', T.basis('spline', degree=degree), X.basis('std', degree=degree)) * ref_mag_pot
        ns.beta2 = function.field('beta2', T.basis('spline', degree=degree), X.basis('std', degree=degree)) / (ref_field_strength / ref_length)
        ns.B_i = 'D_ij grad_j(A2)'
        ns.Bmagsqr = 'B_i B_i'
        ns.Bmag = 'sqrt(abs(Bmagsqr))'
        #ns.nu = numpy.choose(X.indicator('iron'), [1 / ns.mu0 * Permeability('H/m'), nuiron(ns.Bmagsqr) * Permeability('H/m')]) / Permeability('H/m')
        ns.nu = (1 - X.indicator('iron')) / ns.mu0 + X.indicator('iron') * nuiron(ns.Bmagsqr)
        ns.stress_ij = 'nu (B_i B_j - δ_ij B_k B_k / 2)'

        ac_groups = [
            f'ac_{i}_{s}'
            for i in range(geometry.n_stator_teeth_div_mult)
            for s in 'pn'
        ]
        ac_centers = X.locate(
            ns.x0,
            function.eval(numpy.stack([
                X[k].integral('x0_i dV0' @ ns, degree=2) / X[k].integral('dV0' @ ns, degree=2)
                for k in ac_groups
            ])),
            tol=1e-6 * ref_length,
        )
        A2_ac_centers = numpy.reshape(ac_centers.bind(ns.A2), (-1, 3, 2))
        ns.flux = numpy.sum(A2_ac_centers[:,:,0] - A2_ac_centers[:,:,1], axis=0) * geometry.stack_length * geometry.structural_multiplicity

        ns.emf_i = 'dt(flux_i) nturnsac'

        ns.densdc = geometry.n_turns_ac / X['dc_0_p'].integral(ns.dV0, degree=2)
        ns.densac = geometry.n_turns_ac / X['ac_0_p'].integral(ns.dV0, degree=2)
        ns.Rdc = geometry.dc_resistance
        ns.Idc = driver.dc_current
        ns.Udc = 'Rdc Idc'
        ns.Rac = geometry.ac_resistance
        ns.Iacpeak = driver.peak_ac_current
        ns.Iac = driver.ac_current(geometry, ns.thetam)
        ns.Uac_i = 'Iac_i Rac + emf_i'
        ns.J = '(δp - δn) (δdc Idc densdc + δac_i Iac_i densac)'

        #ns.torque = X['air_gap'].integral('B_i nr_i B_j nphi_j r dV / mu0' @ ns, degree=2 * degree) * geometry.stack_length * geometry.structural_multiplicity  / geometry.height_air_gap
        #ns.torque = X['air_gap'].integral('stress_ij nr_i nphi_j r dV' @ ns, degree=2 * degree) * geometry.stack_length * geometry.structural_multiplicity  / geometry.height_air_gap
        ns.torque = X['air_gap'].integral('D_ij x_i stress_jk nr_k dV' @ ns, degree=2 * degree) * geometry.stack_length * geometry.structural_multiplicity  / geometry.height_air_gap

        ns.λ = function.field('λ', T.basis('spline', degree=degree), X.boundary['interface_stator'].basis('std', degree=degree))

        ns.Tavg = lambda q: T.integral(q * ns.dThetam, degree=3 * degree) / T.integral(ns.dThetam, degree=3 * degree)
        ns.Tstddev = lambda q: numpy.sqrt(ns.Tavg((q - ns.Tavg(q))**2))

        ns.avgtorque = 'Tavg(torque)'
        ns.stddevtorque = 'Tstddev(torque)'

        ns.Pac = 'Iac_i Uac_i'
        ns.Pdc = 'Idc Udc'
        ns.Protor = 'torque omegam / rad'

        # INTERFACE BETWEEN STATOR AND ROTOR

        rotation_interface = (T * X.boundary['interface_stator']).sample('gauss', 2 * degree)
        if geometry.structural_multiplicity == 1:
            geom = numpy.concatenate([ns.x / ref_length, ns.thetam[None] / 'rad'])
        else:
            geom = numpy.stack([
                ((ns.phi - ns.thetam + Angle('90deg')) % (Angle('360deg') / geometry.structural_multiplicity)) / 'rad',
                ns.thetam / 'rad',
            ])
        opp = (T * X.boundary['interface_rotor']).locate(geom, rotation_interface.eval(geom), eps=1e-9)
        rotation_interface = rotation_interface.zip(opp.rename_spaces({k: '~'+k for k in 'XT'}))

        # WEAK FORM

        residual = (T * X).integral('(-grad_i(beta2) grad_i(A2) nu + beta2 J) dV dThetam' @ ns, degree=2 * degree) / (ref_area * thetamperiod)
        residual += function.linearize(
            rotation_interface.integral('λ [A2] dS dThetam' @ ns) / (ref_length * ref_mag_pot * thetamperiod),
            'A2:beta2,λ:gamma',
        )
        trials = ['A2', 'λ']
        tests = ['beta2', 'gamma']

        sqr = (T * X.boundary['outer,inner']).integral('A2^2 dS dThetam' @ ns, degree=degree * 2 + 1) / (ref_mag_pot**2 * ref_surf * thetamperiod)
        constraints = System(sqr, trial='A2').solve_constraints(droptol=1e-15)

        assert numpy.ndim(residual) == 0
        system = System(residual, trial=trials, test=tests)

        # VARIABLES TO KEEP

        self.geometry = geometry
        self.T = T
        self.X = X
        self.nthetamperiods = nthetamperiods
        self.thetamperiod = thetamperiod
        self.ns = ns
        self.residual = residual
        self.constraints = constraints
        self.trials = tuple(trials)
        self.tests = tuple(tests)
        self.system = system

    @treelog.withcontext
    def solve(self, *, arguments=None, tol=1e-7):
        return self.system.solve(constrain=self.constraints, arguments=arguments or {}, tol=tol)

    @treelog.withcontext
    def jacobian(self, f, y, arguments):
        arg_objs = function.arguments_for(self.residual)
        if isinstance(y, str):
            y = arg_objs[y]

        R = function.linearize(self.residual, {trial: f'd{trial}dy' for trial in self.trials})

        for trial in self.trials:
            test_arg = arg_objs[f'{trial}test']
            R += numpy.sum(test_arg * function.Argument(f'dR{trial}dy', shape=test_arg.shape))
        system = System(R, trial=tuple(f'd{trial}dy' for trial in self.trials), test=tuple(f'{trial}test' for trial in self.trials))

        constraints = dict(self.constraints)
        for trial in self.trials:
            if trial in constraints:
                c = numpy.copy(constraints[trial])
                c[~numpy.isnan(c)] = 0.
                constraints[f'd{trial}dy'] = c

        dRdy = dict(zip(
            [f'dR{trial}dy' for trial in self.trials],
            function.eval([function.derivative(function.derivative(self.residual, arg_objs[f'{trial}test']), y) for trial in self.trials], arguments),
        ))
        dtrialdy = {f'd{trial}dy': numpy.zeros(arg_objs[trial].shape + y.shape) for trial in self.trials}
        assert y.ndim == 1
        for i in range(y.shape[0]):
            a = system.solve(arguments=arguments | {k: v[...,i] for k, v in dRdy.items()}, constrain=constraints)
            for trial in self.trials:
                dtrialdy[f'd{trial}dy'][...,i] = a[f'd{trial}dy']

        dfdy = function.derivative(f, y)
        for trial in self.trials:
            dfdy += numpy.sum(
                function.derivative(f, arg_objs[trial])[(...,)+(None,)*y.ndim] * function.Argument(f'd{trial}dy', arg_objs[trial].shape + y.shape)[(None,)*f.ndim],
                axis=tuple(range(f.ndim, f.ndim+arg_objs[trial].ndim)),
            )
        return function.eval(dfdy, arguments | dtrialdy)

    def plot_mesh(self):
        with export.mplfigure('mesh.png') as fig:
            fig.suptitle('mesh and subdomains')
            ax = fig.add_axes((0.1, 0.25, 0.8, 0.65))

            handles = []
            labels = []
            colors = iter(matplotlib.colormaps['tab20'].colors)
            for group in 'ac_a_p', 'ac_a_n' ,'ac_b_p', 'ac_b_n', 'ac_c_p', 'ac_c_n', 'dc_p', 'dc_n', 'iron':
                color = next(colors)
                cmap = matplotlib.colors.ListedColormap(color)
                handles.append(matplotlib.patches.Patch(color=color))
                labels.append(group)
                smpl = self.X[group].sample('bezier', 5)
                xsmpl = smpl.eval(self.ns.x0)
                export.triplot(ax, xsmpl/plt_length, tri=smpl.tri, values=numpy.ones(len(xsmpl)), cmap=cmap)

            smpl = self.X.sample('bezier', 5)
            xsmpl = smpl.eval(self.ns.x0)
            export.triplot(ax, xsmpl/plt_length, hull=smpl.hull, plabel=f'[{plt_length}]')

            ax.set_xlabel(f'x_0 [{plt_length}]')
            ax.set_ylabel(f'x_1 [{plt_length}]')
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
            ax.set_aspect('equal')
            ax.autoscale(enable=True, axis='both', tight=True)

    def plot_thetam(self, v, vunit=None, *, title=None, arguments=None, phase=False):
        vlabel = None

        if isinstance(v, str):
            if title is None:
                title = v
            vlabel = v
            v @= self.ns

        if vunit is not None:
            v /= vunit
            if vlabel is None:
                vlabel = f'[{vunit}]'
            else:
                vlabel = f'{vlabel} [{vunit}]'

        if phase:
            desired_shape = 3,
        else:
            desired_shape = ()
        if v.shape != desired_shape:
            raise ValueError(f'expected an `v` with shape {desired_shape} but got {v.shape}')

        if title is None:
            raise ValueError('either `v` must be a `str` or `title` must be specified')

        Tsmpl = self.T.sample('bezier', 17)
        tri = Tsmpl.tri
        thetam, v = Tsmpl.eval([self.ns.thetam / plt_angle, v], arguments or {})

        if phase:

            n = self.nthetamperiods // self.geometry.n_rotor_teeth_div_mult
            tri = numpy.concatenate([tri + i * Tsmpl.npoints for i in range(n)], axis=0)
            thetam = numpy.concatenate([thetam + i * self.thetamperiod / plt_angle for i in range(n)], axis=0)
            v = numpy.stack([
                numpy.concatenate([(1 - 2 * (i % 2)) * v[:,(i + j) % 3] for i in range(n)], axis=0)
                for j in range(3)
            ])

            with export.mplfigure(f'{title}.svg') as fig:
                ax = fig.add_subplot(1, 1, 1)
                labels = []
                for j, phase, color in zip(range(3), 'abc', matplotlib.colormaps['tab10'].colors):
                    lc = ax.add_collection(LineCollection(numpy.asarray([thetam, v[j]]).T[tri], colors=color))
                    labels.append((lc, f'phase {phase}'))
                ax.legend(*zip(*labels))
                ax.set_xlabel(f'$\\theta_m$ [{plt_angle}]')
                ax.set_ylabel(vlabel)
                ax.autoscale(enable=True, axis='x', tight=True)
                ax.autoscale(enable=True, axis='y', tight=False)

        else:

            export.triplot(
                f'{title}.svg',
                thetam[:,None],
                v,
                tri=tri,
                plabel=f'$\\theta_m$ [{plt_angle}]',
                vlabel=vlabel,
            )

    def plot_field(self, v, vunit=None, *, thetam, vlabel=None, title=None, clim=None, arguments=None, contours=None, **kwargs):
        xunit = 'cm'
        if isinstance(v, str):
            if vlabel is None:
                vlabel = v
            if title is None:
                title = v
            v = v @ self.ns
        Tsmpl = self.T.locate(self.ns.thetam / 'rad', numpy.array([thetam / 'rad']), eps=1e-10)
        Xsmpl = self.X.sample('bezier', 5)
        x, v = (Tsmpl * Xsmpl).eval([self.ns.x / xunit, v], arguments or None)
        if vunit is not None:
            v /= vunit
        if isinstance(clim, str) and clim == 'symmetric':
            vmax = numpy.max(abs(v))
            clim = -vmax, vmax
        with nutils.export.mplfigure(f'{title}.png') as fig:
            ax = fig.add_subplot(1, 1, 1)
            im = nutils.export.triplot(
                ax,
                x,
                v,
                tri=Xsmpl.tri,
                clim=clim,
                **kwargs,
            )
            if contours is not None:
                if isinstance(contours, str):
                    contours @= self.ns
                if isinstance(contours, Quantity):
                    contours = contours.unwrap()
                contours = (Tsmpl * Xsmpl).eval(contours, arguments or {})
                ax.tricontour(
                    x.T[0],
                    x.T[1],
                    Xsmpl.tri,
                    contours,
                    levels=numpy.linspace(numpy.min(contours), numpy.max(contours), num=14),
                    colors='w',
                    linestyles='solid',
                    linewidths=.9,
                )
            ax.set_xlabel(f'$x_0$ [{xunit}]')
            ax.set_ylabel(f'$x_1$ [{xunit}]')
            fig.colorbar(im, label=vlabel if vunit is None else f'{vlabel or ""} [{vunit}]')
            if title is not None:
                fig.suptitle(title)

    def print_scalar(self, expr, unit=None, arguments=None):
        v = nutils.function.eval(expr @ self.ns, arguments or {})
        treelog.user((f'{expr}: {{:.3{unit}}}' if unit else f'{expr}: {{}}').format(v))


def main(
    geometry: Geometry = Geometry(),
    driver: Driver = Driver(),
    nelems_angle: int = 6,
    degree: int = 2,
):
    machine = Machine(geometry, driver, nelems_angle, degree)
    arguments = machine.solve()
    machine.plot_mesh()
    machine.plot_field('sqrt(B_i B_i)', 'T', arguments=arguments, contours='A2', angle=Angle('0deg'))
    machine.print_scalar('Tavg(torque)', 'N*m', arguments=arguments)


def cos(angle):
    return numpy.cos(angle / 'rad')

def sin(angle):
    return numpy.sin(angle / 'rad')

def arctan2(a, b):
    if isinstance(a, Quantity) or isinstance(b, Quantity):
        if type(a) != type(b):
            raise ValueError('arguments have different types: {a}, {b}')
        a = a.unwrap()
        b = b.unwrap()
    return numpy.arctan2(a, b) * Angle('rad')

# TEMPORARY FIX

from nutils import evaluable
if evaluable.Mod._derivative is evaluable.Pointwise._derivative:
    def derivative(self, var, seen):
        if self.dtype == float and evaluable.iszero(evaluable.derivative(self.divisor, var, seen)):
            return evaluable.derivative(self.dividend, var, seen)
        return super()._derivative(var, seen)
    evaluable.Mod._derivative = derivative
    del derivative

if __name__ == '__main__':
    from nutils import cli
    cli.run(main)
