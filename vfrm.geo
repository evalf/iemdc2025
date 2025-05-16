period = 2*Pi/m;

// derived quantities
r_ry = r_ri + h_ry; // rotor yoke radius
r_rt = r_ry + h_rt; // rotor teeth radius
r_i = r_rt + f_i * h_a; // radius of interface between rotor and stator sections
r_ao = r_rt + h_a; // air gap outer radius
r_si = r_ao; // stator inner radius
r_so = r_si + h_s; // stator outer radius
a_sci = period * r_si / n_st - a_sti;

// center point
p0 = newp; Point(p0) = {0, 0, 0}; // center

// rotor
For i In {0:n_rt-1}
    p_r[7*i+0] = newp; a = period*i/n_rt;                    r = r_ri; Point(p_r[7*i+0]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*i+1] = newp; a = period*i/n_rt;                    r = r_rt; Point(p_r[7*i+1]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*i+2] = newp; a = period*i/n_rt;                    r = r_i;  Point(p_r[7*i+2]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*i+3] = newp; a = period*i/n_rt+a_rto/2/r_rt;       r = r_rt; Point(p_r[7*i+3]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*i+4] = newp; a = period*(i+0.5)/n_rt-a_rti/2/r_ry; r = r_ry; Point(p_r[7*i+4]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*i+5] = newp; a = period*(i+0.5)/n_rt+a_rti/2/r_ry; r = r_ry; Point(p_r[7*i+5]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*i+6] = newp; a = period*(i+1)/n_rt-a_rto/2/r_rt;   r = r_rt; Point(p_r[7*i+6]) = {r*Cos(a), r*Sin(a), 0};
    l_r01[i] = newc; Line(l_r01[i]) = {p_r[7*i+0], p_r[7*i+1]};
    l_r12[i] = newc; Line(l_r12[i]) = {p_r[7*i+1], p_r[7*i+2]};
EndFor
If (m == 1)
    p_r[7*n_rt+0] = p_r[0];
    p_r[7*n_rt+1] = p_r[1];
    p_r[7*n_rt+2] = p_r[2];
    l_r01[n_rt] = l_r01[0];
    l_r12[n_rt] = l_r12[0];
Else
    p_r[7*n_rt+0] = newp; a = period; r = r_ri; Point(p_r[7*n_rt+0]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*n_rt+1] = newp; a = period; r = r_rt; Point(p_r[7*n_rt+1]) = {r*Cos(a), r*Sin(a), 0};
    p_r[7*n_rt+2] = newp; a = period; r = r_i;  Point(p_r[7*n_rt+2]) = {r*Cos(a), r*Sin(a), 0};
    l_r01[n_rt] = newc; Line(l_r01[n_rt]) = {p_r[7*n_rt+0], p_r[7*n_rt+1]};
    l_r12[n_rt] = newc; Line(l_r12[n_rt]) = {p_r[7*n_rt+1], p_r[7*n_rt+2]};
    Periodic Curve {l_r01[n_rt]} = {l_r01[0]} Rotate {{0,0,1},{0,0,0},period};
    Periodic Curve {l_r12[n_rt]} = {l_r12[0]} Rotate {{0,0,1},{0,0,0},period};
EndIf
For i In {0:n_rt-1}
    l_01 = l_r01[i];
    l_12 = l_r12[i];
    l_13 = newc; Circle(l_13) = {p_r[7*i+1], p0, p_r[7*i+3]};
    l_34 = newc; Line(l_34) = {p_r[7*i+3], p_r[7*i+4]};
    l_45 = newc; Circle(l_45) = {p_r[7*i+4], p0, p_r[7*i+5]};
    l_56 = newc; Line(l_56) = {p_r[7*i+5], p_r[7*i+6]};
    l_68 = newc; Circle(l_68) = {p_r[7*i+6], p0, p_r[7*i+8]};
    l_36 = newc; Circle(l_36) = {p_r[7*i+3], p0, p_r[7*i+6]};
    l_07 = newc; Circle(l_07) = {p_r[7*i+0], p0, p_r[7*i+7]};
    l_29 = newc; Circle(l_29) = {p_r[7*i+2], p0, p_r[7*i+9]};
    l_78 = l_r01[i+1];
    l_89 = l_r12[i+1];
    l_ri[] += {l_07};
    b = newcl; Curve Loop(b) = {l_01, l_13, l_34, l_45, l_56, l_68, -l_78, -l_07}; s = news; Plane Surface(s) = {b};
    s_ri[] += {s};
    s_rs[] += {s};
    b = newcl; Curve Loop(b) = {l_36, -l_56, -l_45, -l_34}; s = news; Plane Surface(s) = {b};
    s_rs[] += {s};

    If (f_i > 0)
        l_ir[] += {l_29};
        b = newcl; Curve Loop(b) = {l_13, l_36, l_68, l_89, -l_29, -l_12}; s = news; Plane Surface(s) = {b};
        s_rs[] += {s};
        s_ag[] += {s};
    Else
        l_ir[] += {l_13, l_36, l_68};
    EndIf
EndFor
Physical Surface("rotor_iron") = {s_ri[]};
Physical Surface("rotor_section") = {s_rs[]};
Physical Surface("rotating") = {s_rs[]};
Physical Curve("inner") = {l_ri[]};
Physical Curve("interface_rotor") = {l_ir[]};

// stator
For i In {0:n_st-(m == 1)}
    a = period*(i-0.5)/n_st;
    p_sb[5*i+0] = newp; r = r_i;              Point(p_sb[5*i+0]) = {r*Cos(a), r*Sin(a), 0};
    p_sb[5*i+1] = newp; r = r_si;             Point(p_sb[5*i+1]) = {r*Cos(a), r*Sin(a), 0};
    p_sb[5*i+2] = newp; r = r_si+h_sca;       Point(p_sb[5*i+2]) = {r*Cos(a), r*Sin(a), 0};
    p_sb[5*i+3] = newp; r = r_si+h_sca+h_scd; Point(p_sb[5*i+3]) = {r*Cos(a), r*Sin(a), 0};
    p_sb[5*i+4] = newp; r = r_so;             Point(p_sb[5*i+4]) = {r*Cos(a), r*Sin(a), 0};
    For j In {0:3}
        l_sb[4*i+j] = newc; Line(l_sb[4*i+j]) = {p_sb[5*i+j+0], p_sb[5*i+j+1]};
    EndFor
EndFor
If (m == 1)
    For j In {0:4}
        p_sb[5*n_st+j] = p_sb[j];
    EndFor
    For j In {0:3}
        l_sb[4*n_st+j] = l_sb[j];
    EndFor
Else
    For j In {0:3}
        Periodic Curve{l_sb[4*n_st+j]} = {l_sb[j]} Rotate {{0,0,1},{0,0,0},period};
    EndFor
EndIf
For i In {0:n_st-1}
    p_a = p_sb[5*i+0];
    p_b = p_sb[5*i+1];
    p_c = p_sb[5*i+2];
    p_d = p_sb[5*i+3];
    p_e = p_sb[5*i+4];

    a = period*i/n_st-a_sti/(2*r_si);
    p_f = newp; r = r_si;             Point(p_f) = {r*Cos(a), r*Sin(a), 0};
    p_g = newp; r = r_si+h_sca;       Point(p_g) = {r*Cos(a), r*Sin(a), 0};
    p_h = newp; r = r_si+h_sca+h_scd; Point(p_h) = {r*Cos(a), r*Sin(a), 0};
    a = period*i/n_st+a_sti/(2*r_si);
    p_i = newp; r = r_si;             Point(p_i) = {r*Cos(a), r*Sin(a), 0};
    p_j = newp; r = r_si+h_sca;       Point(p_j) = {r*Cos(a), r*Sin(a), 0};
    p_k = newp; r = r_si+h_sca+h_scd; Point(p_k) = {r*Cos(a), r*Sin(a), 0};

    p_l = p_sb[5*(i+1)+0];
    p_m = p_sb[5*(i+1)+1];
    p_n = p_sb[5*(i+1)+2];
    p_o = p_sb[5*(i+1)+3];
    p_p = p_sb[5*(i+1)+4];

    l_ab = l_sb[4*i+0];
    l_bc = l_sb[4*i+1];
    l_cd = l_sb[4*i+2];
    l_de = l_sb[4*i+3];
    l_fg = newc; Line(l_fg) = {p_f, p_g};
    l_gh = newc; Line(l_gh) = {p_g, p_h};
    l_ij = newc; Line(l_ij) = {p_i, p_j};
    l_jk = newc; Line(l_jk) = {p_j, p_k};
    l_lm = l_sb[4*(i+1)+0];
    l_mn = l_sb[4*(i+1)+1];
    l_no = l_sb[4*(i+1)+2];
    l_op = l_sb[4*(i+1)+3];
    l_al = newc; Circle(l_al) = {p_a, p0, p_l};
    l_bf = newc; Circle(l_bf) = {p_b, p0, p_f};
    l_cg = newc; Line(l_cg) = {p_c, p_g};
    l_dh = newc; Line(l_dh) = {p_d, p_h};
    l_ep = newc; Circle(l_ep) = {p_e, p0, p_p};
    l_fi = newc; Circle(l_fi) = {p_f, p0, p_i};
    l_im = newc; Circle(l_im) = {p_i, p0, p_m};
    l_jn = newc; Line(l_jn) = {p_j, p_n};
    l_ko = newc; Line(l_ko) = {p_k, p_o};

    l_so[] += {l_ep};

    b = newcl; Curve Loop(b) = {l_de, l_ep, -l_op, -l_ko, -l_jk, -l_ij, -l_fi, l_fg, l_gh, -l_dh}; s = news; Plane Surface(s) = {b};
    s_si[] += {s};
    s_ss[] += {s};

    b = newcl; Curve Loop(b) = {l_bc, l_cg, -l_fg, -l_bf}; s_ac_p = news; Plane Surface(s_ac_p) = {b};
    b = newcl; Curve Loop(b) = {l_cd, l_dh, -l_gh, -l_cg}; s_dc_0 = news; Plane Surface(s_dc_0) = {b};
    b = newcl; Curve Loop(b) = {l_ij, l_jn, -l_mn, -l_im}; s_ac_n = news; Plane Surface(s_ac_n) = {b};
    b = newcl; Curve Loop(b) = {l_jk, l_ko, -l_no, -l_jn}; s_dc_1 = news; Plane Surface(s_dc_1) = {b};
    s_ss[] += {s_ac_p, s_ac_n, s_dc_0, s_dc_1};
    Physical Surface(Sprintf("ac_%g_p", i)) = {s_ac_p};
    Physical Surface(Sprintf("ac_%g_n", i)) = {s_ac_n};
    If (i % 2 == 0)
        Physical Surface(Sprintf("dc_%g_p", i)) = {s_dc_1};
        Physical Surface(Sprintf("dc_%g_n", i)) = {s_dc_0};
    Else
        Physical Surface(Sprintf("dc_%g_p", i)) = {s_dc_0};
        Physical Surface(Sprintf("dc_%g_n", i)) = {s_dc_1};
    EndIf

    If (f_i < 1)
        l_is[] += {l_al};
        b = newc; Curve Loop(b) = {l_ab, l_bf, l_fi, l_im, -l_lm, -l_al}; s = news; Plane Surface(s) = {b};
        s_ss[] += {s};
        s_ag[] += {s};
    Else
        l_is[] += {l_bf, l_fi, l_im};
    EndIf
EndFor
Physical Surface("stator_iron") = {s_si[]};
Physical Surface("stator_section") = {s_ss[]};
Physical Curve("outer") = {l_so[]};
Physical Curve("interface_stator") = {l_is[]};

Physical Surface("air_gap") = {s_ag[]};

// element size
Field[1] = MathEval;
Field[1].F = Sprintf("Abs(Sqrt(x*x+y*y) - %g)", (r_rt + r_si) / 2);

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_a / n_ea;
Field[2].SizeMax = h_e;
Field[2].DistMin = h_a / 2;
Field[2].DistMax = h_a / 2 + h_e;

Background Field = 2;
