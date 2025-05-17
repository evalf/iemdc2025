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
from nutils.SI import Area, CurrentDensity, Dimension, ElectricCurrent, ElectricPotential, Force, Length, MagneticFieldStrength, MagneticFluxDensity, Resistance, Time, Quantity
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

μ0 = 4e-7 * numpy.pi * Permeability('H/m')

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
        topo, geom = mesh.gmsh(
            Path(__file__).parent/'vfrm.geo',
            dimension=2,
            order=2,
            space=space,
            numbers=dict(
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
            ),
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

        # Plot the mesh and subdomain.
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
                smpl = topo[group].sample('bezier', 5)
                xsmpl = smpl.eval(geom)
                export.triplot(ax, xsmpl/plt_length, tri=smpl.tri, values=numpy.ones(len(xsmpl)), cmap=cmap)

            smpl = topo.sample('bezier', 5)
            xsmpl = smpl.eval(geom)
            export.triplot(ax, xsmpl/plt_length, hull=smpl.hull, plabel=f'[{plt_length}]')

            ax.set_xlabel(f'x_0 [{plt_length}]')
            ax.set_ylabel(f'x_1 [{plt_length}]')
            ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
            ax.set_aspect('equal')
            ax.autoscale(enable=True, axis='both', tight=True)

        return topo, geom


@dataclass(frozen=True)
class Driver:
    peak_ac_current: ElectricCurrent = ElectricCurrent('28.868A')
    dc_current: ElectricCurrent = ElectricCurrent('50A')
    commutation_angle: Angle = Angle('90deg')
    mechanical_angular_frequency: AngularFrequency = AngularFrequency('1000rpm')

    def ac_current(self, geometry: Geometry, mechanical_angle):
        θe = geometry.n_rotor_teeth * mechanical_angle - self.commutation_angle + numpy.stack([0, -120, 120]) * Angle('deg')
        return self.peak_ac_current * cos(θe)


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
        ref_field_strength = ref_flux_dens / μ0
        ref_elec_pot = ElectricPotential('V')

        # ROTATION MESH

        Θ, θm = mesh.line(nelems_angle, space=f'Θ', bnames=['begin', 'end'])
        nΘmperiods = geometry.n_stator_teeth_div_mult * geometry.n_rotor_teeth_div_mult
        Θmperiod = Angle('360deg') / geometry.structural_multiplicity / nΘmperiods
        θm *= Θmperiod / Θ.shape[0]

        # SPACE MESH

        X, x0 = geometry.mesh(space='X')
        xrot = numpy.stack([cos(θm) * x0[0] - sin(θm) * x0[1], sin(θm) * x0[0] + cos(θm) * x0[1]])
        x = numpy.choose(X.indicator('rotating'), [x0 / ref_length, xrot / ref_length]) * ref_length

        # MAGNETIC RELUCTIVITY OF IRON

        νiron = lambda Bmagsqr: 1 / (5000 * μ0)
        #νiron = lambda Bmagsqr: numpy.exp(4.41797133 + 1.16730888 * Bmagsqr / 'T2') / Permeability('T*m/A')

        #Hiron, Biron = numpy.loadtxt('BHcurve_NO27.txt').T
        #Hiron *= MagneticFieldStrength('A/m')
        #Biron *= MagneticFluxDensity('T')
        #νiron = lambda Bmagsqr: numpy.interp(Bmagsqr, Biron[1:]**2, Hiron[1:] / Biron[1:])

        # NAMESPACE

        ns = Namespace()
        ns.cos = cos
        ns.sin = sin
        ns.δ = function.eye(2)
        ns.D = function.levicivita(2)
        ns.μ0 = μ0
        ns.ωm = driver.mechanical_angular_frequency
        ns.nturnsac = geometry.n_turns_ac
        ns.nturnsdc = geometry.n_turns_dc
        ns.rad = Angle('rad')

        ns.δp = X.indicator('p')
        ns.δn = X.indicator('n')
        ns.δdc = X.indicator('dc')
        ns.δac = numpy.stack([X.indicator(f'ac_{k}') for k in 'abc'])

        ns.θm = θm
        ns.t = θm / driver.mechanical_angular_frequency
        ns.x0 = x0
        ns.x = x
        ns.r = numpy.linalg.norm(ns.x)
        ns.φ = arctan2(ns.x[1], ns.x[0])
        ns.define_for('θm', gradient='∂θm', jacobians=('dΘm',))
        ns.define_for('t', gradient='∂t', jacobians=('dt',))
        ns.define_for('x0', gradient='∇0', jacobians=('dV0', 'dS0'), normal='n0', spaces=X.spaces)
        ns.define_for('x', gradient='∇', jacobians=('dV', 'dS'), normal='n', spaces=X.spaces)
        ns.nr_i = 'x_i / r'
        ns.nφ_i = 'D_ji nr_j'

        ns.A = function.field('A', Θ.basis('spline', degree=degree), X.basis('std', degree=degree)) * ref_mag_pot
        ns.Atest = function.field('Atest', Θ.basis('spline', degree=degree), X.basis('std', degree=degree)) / (ref_field_strength / ref_length)
        ns.B_i = 'D_ij ∇_j(A)'
        ns.Bmagsqr = 'B_i B_i'
        ns.Bmag = 'sqrt(abs(Bmagsqr))'
        #ns.ν = numpy.choose(X.indicator('iron'), [1 / ns.μ0 * Permeability('H/m'), νiron(ns.Bmagsqr) * Permeability('H/m')]) / Permeability('H/m')
        ns.ν = (1 - X.indicator('iron')) / ns.μ0 + X.indicator('iron') * νiron(ns.Bmagsqr)
        ns.σ_ij = 'ν (B_i B_j - δ_ij B_k B_k / 2)'

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
        A_ac_centers = numpy.reshape(ac_centers.bind(ns.A), (-1, 3, 2))
        ns.Φ = numpy.sum(A_ac_centers[:,:,0] - A_ac_centers[:,:,1], axis=0) * geometry.stack_length * geometry.structural_multiplicity

        ns.emf_i = '∂t(Φ_i) nturnsac'

        ns.densdc = geometry.n_turns_ac / X['dc_0_p'].integral(ns.dV0, degree=2)
        ns.densac = geometry.n_turns_ac / X['ac_0_p'].integral(ns.dV0, degree=2)
        ns.Idc = driver.dc_current
        ns.Udc = ns.Idc * geometry.dc_resistance
        ns.Iac = driver.ac_current(geometry, ns.θm)
        ns.Uac = ns.Iac * geometry.ac_resistance + ns.emf
        ns.J = '(δp - δn) (δdc Idc densdc + δac_i Iac_i densac)'

        #ns.τ = X['air_gap'].integral('B_i nr_i B_j nφ_j r dV / μ0' @ ns, degree=2 * degree) * geometry.stack_length * geometry.structural_multiplicity  / geometry.height_air_gap
        #ns.τ = X['air_gap'].integral('σ_ij nr_i nφ_j r dV' @ ns, degree=2 * degree) * geometry.stack_length * geometry.structural_multiplicity  / geometry.height_air_gap
        ns.τ = X['air_gap'].integral('D_ij x_i σ_jk nr_k dV' @ ns, degree=2 * degree) * geometry.stack_length * geometry.structural_multiplicity  / geometry.height_air_gap

        ns.λ = function.field('λ', Θ.basis('spline', degree=degree), X.boundary['interface_stator'].basis('std', degree=degree))

        ns.Θmavg = lambda q: Θ.integral(q * ns.dΘm, degree=2 * degree) / Θ.integral(ns.dΘm, degree=2 * degree)
        ns.Θmstddev = lambda q: numpy.sqrt(ns.Θmavg((q - ns.Θmavg(q))**2))
        ns.tavg = lambda q: Θ.integral(q * ns.dt, degree=2 * degree) / Θ.integral(ns.dt, degree=2 * degree)

        ns.τavg = 'Θmavg(τ)'
        ns.τstddev = 'Θmstddev(τ)'

        ns.Pac = 'tavg(Iac_i Uac_i)'
        #ns.Sac = 'tavg(Iac_i Uac_i)'
        #ns.Pac = ('tavg(Iac_i Iac_i)' @ ns) * geometry.ac_resistance
        ns.Pdc = 'Idc Udc'
        ns.Protor = 'τavg ωm / rad'

        # INTERFACE BETWEEN STATOR AND ROTOR

        rotation_interface = (Θ * X.boundary['interface_stator']).sample('gauss', 2 * degree)
        if geometry.structural_multiplicity == 1:
            geom = numpy.concatenate([ns.x / ref_length, ns.θm[None] / 'rad'])
        else:
            geom = numpy.stack([
                ((ns.φ - ns.θm + Angle('90deg')) % (Angle('360deg') / geometry.structural_multiplicity)) / 'rad',
                ns.θm / 'rad',
            ])
        opp = (Θ * X.boundary['interface_rotor']).locate(geom, rotation_interface.eval(geom), eps=1e-9)
        rotation_interface = rotation_interface.zip(opp.rename_spaces({k: '~'+k for k in 'XΘ'}))

        # WEAK FORM

        residual = (Θ * X).integral('(-∇_i(Atest) ∇_i(A) ν + Atest J) dV dΘm' @ ns, degree=2 * degree) / (ref_area * Θmperiod)
        residual += function.linearize(
            rotation_interface.integral('λ [A] dS dΘm' @ ns) / (ref_length * ref_mag_pot * Θmperiod),
            'λ:λtest,A:Atest',
        )
        trials = ['A', 'λ']

        sqr = (Θ * X.boundary['outer,inner']).integral('A^2 dS dΘm' @ ns, degree=degree * 2 + 1) / (ref_mag_pot**2 * ref_surf * Θmperiod)
        constraints = System(sqr, trial='A').solve_constraints(droptol=1e-15)

        assert numpy.ndim(residual) == 0
        system = System(residual, trial=trials, test=tuple(trial+'test' for trial in trials))

        # VARIABLES TO KEEP

        self.geometry = geometry
        self.Θ = Θ
        self.X = X
        self.nΘmperiods = nΘmperiods
        self.Θmperiod = Θmperiod
        self.ns = ns
        self.residual = residual
        self.constraints = constraints
        self.trials = tuple(trials)
        self.tests = tuple(trial + 'test' for trial in trials)
        self.system = system

    @treelog.withcontext
    def solve(self, arguments=None, /, *, tol=1e-7):
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

    def plot(self, arguments, /):
        τavg, τstddev, Pac, Pdc, Protor = function.eval(['τavg', 'τstddev', 'Pac', 'Pdc', 'Protor'] @ self.ns, arguments)
        treelog.user(f'average torque: {τavg:0.3N*m}')
        treelog.user(f'stddev torque: {τstddev:0.3N*m}')
        treelog.user(f'ac power: {Pac:0.3W}')
        treelog.user(f'dc power: {Pdc:0.3W}')
        treelog.user(f'rotor power: {Protor:0.3W}')

        Θsmpl = self.Θ.sample('bezier', 17)
        funcs = 'θm' , 'τ', 'Φ_i', 'Uac_i', 'Iac_i', 'emf_i'
        smpld = dict(zip(funcs, Θsmpl.eval(funcs @ self.ns, arguments)))
        export.triplot('torque.svg', smpld['θm'][:,None] / plt_angle, smpld['τ'] / plt_torque, tri=Θsmpl.tri, plabel=f'θm [{plt_angle}]', vlabel=f'τ [{plt_torque}]')

        n = self.nΘmperiods // self.geometry.n_rotor_teeth_div_mult
        tri_full = numpy.concatenate([Θsmpl.tri + i * Θsmpl.npoints for i in range(n)], axis=0)
        θm_full = numpy.concatenate([smpld['θm'] + i * self.Θmperiod for i in range(n)], axis=0)
        Φ_0_full = numpy.concatenate([(1 - 2 * (i % 2)) * smpld['Φ_i'][:,i % 3] for i in range(n)], axis=0)
        emf_0_full = numpy.concatenate([(1 - 2 * (i % 2)) * smpld['emf_i'][:,i % 3] for i in range(n)], axis=0)
        Uac_0_full = numpy.concatenate([(1 - 2 * (i % 2)) * smpld['Uac_i'][:,i % 3] for i in range(n)], axis=0)
        Iac_0_full = numpy.concatenate([(1 - 2 * (i % 2)) * smpld['Iac_i'][:,i % 3] for i in range(n)], axis=0)
        export.triplot('Φ.svg', θm_full[:,None] / plt_angle, Φ_0_full / plt_flux, tri=tri_full, plabel=f'θm [{plt_angle}]', vlabel=f'Φ phase a [{plt_flux}]')
        export.triplot('emf.svg', θm_full[:,None] / plt_angle, emf_0_full / plt_elec_pot, tri=tri_full, plabel=f'θm [{plt_angle}]', vlabel=f'emf phase a [{plt_elec_pot}]')
        export.triplot('Uac.svg', θm_full[:,None] / plt_angle, Uac_0_full / plt_elec_pot, tri=tri_full, plabel=f'θm [{plt_angle}]', vlabel=f'U phase a [{plt_elec_pot}]')
        export.triplot('Iac.svg', θm_full[:,None] / plt_angle, Iac_0_full / plt_current, tri=tri_full, plabel=f'θm [{plt_angle}]', vlabel=f'I phase a [{plt_current}]')

        funcs = 'x_i' ,'A', 'Bmag', 'B_i nr_i', 'B_i nφ_i', 'J'
        Θpoints = numpy.linspace(0, self.Θmperiod / Angle('deg'), 7) * Angle('deg')
        Θsmpl = self.Θ.locate(self.ns.θm / 'rad', Θpoints / 'rad', eps=1e-10)
        Xsmpl = self.X.sample('bezier', 5)
        smpld = dict(zip(funcs, function.eval([Θsmpl.bind(Xsmpl.bind(func @ self.ns)) for func in funcs], arguments)))
        plot_B_max = numpy.max(smpld['Bmag'])
        plot_A_max = numpy.max(abs(smpld['A']))
        plot_J_max = numpy.max(abs(smpld['J']))
        for i, θ in enumerate(Θpoints):
            with treelog.context(f'θ = {θ:.1deg}'):
                plot_field('magnetic scalar potential', Xsmpl.tri, smpld['x_i'][i], plt_length, smpld['A'][i], plt_mag_pot, field_lim=(-plot_A_max, plot_A_max))
                plot_field('magnetic flux density (magnitude)', Xsmpl.tri, smpld['x_i'][i], plt_length, smpld['Bmag'][i], plt_flux_dens, field_lim=(MagneticFluxDensity('0T'), plot_B_max))
                plot_field('magnetic flux density radial direction', Xsmpl.tri, smpld['x_i'][i], plt_length, smpld['B_i nr_i'][i], plt_flux_dens, field_lim=(-plot_B_max, plot_B_max), cmap='bwr')
                plot_field('magnetic flux density angular direction', Xsmpl.tri, smpld['x_i'][i], plt_length, smpld['B_i nφ_i'][i], plt_flux_dens, field_lim=(-plot_B_max, plot_B_max), cmap='bwr')
                plot_field('current density', Xsmpl.tri, smpld['x_i'][i], plt_length, smpld['J'][i], plt_cur_dens, field_lim=(-plot_J_max, plot_J_max), cmap='bwr')


def main(
    geometry: Geometry = Geometry(),
    driver: Driver = Driver(),
    nelems_angle: int = 6,
    degree: int = 2,
):
    machine = Machine(geometry, driver, nelems_angle, degree)
    arguments = machine.solve()
    machine.plot(arguments)


def optimize(
    geometry: Geometry = Geometry(),
    driver: Driver = Driver(),
    nelems_angle: int = 6,
    degree: int = 2,
    torque: Torque = Torque('5N*m'),
):
    Iacpeak = function.Argument('Iacpeak', (), float) * ElectricCurrent('A')
    Idc = function.Argument('Idc', (), float) * ElectricCurrent('A')
    arguments = {}
    sqr = ((Iacpeak - driver.peak_ac_current) / 'A')**2
    sqr += ((Idc - driver.dc_current) / 'A')**2
    arguments = System(sqr, 'Iacpeak,Idc').solve(tol=1e-10)
    driver = dataclasses.replace(
        driver,
        peak_ac_current=Iacpeak,
        dc_current=Idc,
    )
    machine = Machine(geometry, driver, nelems_angle, degree)

    arguments = machine.solve(arguments)

    λtorque = function.field('λtorque') / Torque('N*m')

    import nutils_solver_extra
    fun = ('Pac + Pdc' @ machine.ns) / 'W' + λtorque * (machine.ns.τavg - torque)
    arguments = nutils_solver_extra.minimize(
        fun,
        'Iacpeak,Idc,λtorque',
        machine.residual,
        machine.trials,
        machine.tests,
        constrain=machine.constraints,
        arguments=arguments,
        tol=1e-8,
    )

    smpld_Iacpeak, smpld_Idc = function.eval([Iacpeak, Idc], arguments=arguments)
    treelog.user(f'optimal ac peak current: {smpld_Iacpeak:.3A}')
    treelog.user(f'optimal dc current: {smpld_Idc:.3A}')
    machine.plot(arguments)


def plot_field(name, tri, geom, geom_unit, field, field_unit, *, field_lim = None, cmap = None):
    with export.mplfigure(f'{name}.png', dpi=150) as fig:
        fig.suptitle(name)
        ax = fig.add_subplot(1, 1, 1)
        im = export.triplot(ax, geom / geom_unit, field / field_unit, tri=tri, cmap=cmap)
        if field_lim:
            im.set_clim((field_lim[0] / field_unit, field_lim[1] / field_unit))
        fig.colorbar(im, label=f'[{field_unit}]')
        ax.set_xlabel(f'x_0 [{geom_unit}]')
        ax.set_ylabel(f'x_1 [{geom_unit}]')

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
    cli.run(optimize)
