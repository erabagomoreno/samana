from lenstronomy.Util.param_util import shear_cartesian2polar
from lenstronomy.Util.param_util import ellipticity2phi_q
import numpy as np

def get_true_params(lens_index, macromodel_samples_list):

    if lens_index == 1:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.1, 'e2': -0.1,
                            'gamma': 1.8894064923991685, 'a4_a': -0.004488778077207725,
                            'a3_a': -0.004010864193243214, 'delta_phi_m3': -0.08689435347866731,
                            'delta_phi_m4': 0.0}, {'gamma1': -0.04, 'gamma2': 0.04}]
    elif lens_index == 2:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.19537964099282026, 'e2': -0.158688169380922,
          'gamma': 1.892554448155309, 'a4_a': -0.004348301531541436, 'a3_a': 0.00022775720058585444,
          'delta_phi_m3': -0.06702598174099228, 'delta_phi_m4': 0.0}, {'gamma1': -0.06, 'gamma2': 0.03}]
    elif lens_index == 3:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.09805538702908832, 'e2': 0.002701904179452389,
                        'gamma': 1.9539957879337315, 'a4_a': 0.010248068016826362,
                        'a3_a': -0.005147357370025712, 'delta_phi_m3': 0.053195439182018855, 'delta_phi_m4': 0.0}, {'gamma1': -0.01, 'gamma2': 0.09}]
    elif lens_index == 4:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.14902045524131477, 'e2': 0.16189838398964046,
          'gamma': 2.145162560647839, 'a4_a': 0.0004500167278941548, 'a3_a': 0.002251807533098353,
          'delta_phi_m3': 0.48907250375086353, 'delta_phi_m4': 0.0}, {'gamma1': -0.01, 'gamma2': 0.09}]
    elif lens_index == 5:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.0034955619674783118,
                        'e2': 0.12626665005177207, 'gamma': 2.1367585524167274,
                        'a4_a': 0.003715905688783965, 'a3_a': -0.0014697990179525606,
                        'delta_phi_m3': -0.2911280704507563, 'delta_phi_m4': 0.0}, {'gamma1': 0.1, 'gamma2': -0.05}]
    elif lens_index == 6:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.07603400411704919,
                        'e2': 0.24238190469213294, 'gamma': 2.0, 'a4_a': -0.005256260448144398,
                        'a3_a': 0.005024586292580997, 'delta_phi_m3': 0.41140218854650545,
                        'delta_phi_m4': 0.0}, {'gamma1': -0.04, 'gamma2': 0.04}]
    elif lens_index == 7:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.009316606097975175, 'e2': 0.32009147091177126,
          'gamma': 2.0, 'a4_a': 0.014570832287334776, 'a3_a': -0.0016030162747218797,
          'delta_phi_m3': -0.4436889218298895, 'delta_phi_m4': 0.0}, {'gamma1': -0.04, 'gamma2': 0.04}]
    elif lens_index == 8:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.003217299791470716, 'e2': -0.022608223914894157,
          'gamma': 2.0573463026292904, 'a4_a': 0.0018056599088162486, 'a3_a': -0.00782030346109874,
          'delta_phi_m3': 0.3910543561483979, 'delta_phi_m4': 0.0}, {'gamma1': -0.0, 'gamma2': 0.065}]
    elif lens_index == 9:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.1978279163146989, 'e2': 0.08773891128358625,
                        'gamma': 1.959335215553067, 'a4_a': -0.01, 'a3_a': 0.005, 'delta_phi_m3': -0.5127349870534571,
                        'delta_phi_m4': 0.0}, {'gamma1': -0.03, 'gamma2': 0.045}]
    elif lens_index == 10:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.0010025953139360796,
                        'e2': 0.16708081865317795, 'gamma': 1.933562029172763, 'a4_a': -0.001354484915560101,
                        'a3_a': 0.00018886307201135396, 'delta_phi_m3': 0.28412631321802206, 'delta_phi_m4': 0.0},
                       {'gamma1': 0.01, 'gamma2': -0.025}]
    elif lens_index == 11:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.05750376366439197,
                        'e2': 0.16209438330567794, 'gamma': 1.9681100995264538,
                        'a4_a': -0.0038259098896556345, 'a3_a': -0.00014642596529442495, 'delta_phi_m3': -0.33482079885157356, 'delta_phi_m4': 0.0}, {'gamma1': -0.05, 'gamma2': -0.05}]
    elif lens_index == 12:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.03288947350554445,
                        'e2': 0.0022859778522262854, 'gamma': 1.95793079624987, 'a4_a': 0.008941596482682836,
                        'a3_a': -0.004408132271037109, 'delta_phi_m3': -0.3621598245727988, 'delta_phi_m4': 0.0}, {'gamma1': 0.09, 'gamma2': -0.02}]
    elif lens_index == 13:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.22066354572659455, 'e2': -0.09420233055103952, 'gamma': 2.063256852303305, 'a4_a': -0.0053543373280677264, 'a3_a': 0.003307751130197963, 'delta_phi_m3': 0.29080928431429665, 'delta_phi_m4': 0.0}, {'gamma1': 0.01, 'gamma2': -0.02}]
    elif lens_index == 14:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.0017017910032623527, 'e2': 0.059370512275293344,
          'gamma': 2.04195049845922, 'a4_a': 0.0034161376924942296, 'a3_a': 0.0023162406206717074,
          'delta_phi_m3': 0.01460143545425352, 'delta_phi_m4': 0.0}, {'gamma1': 0.1, 'gamma2': 0.09}]
    elif lens_index == 15:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.19107250691324595, 'e2': 0.029535171861108933,
          'gamma': 1.9331820500036354, 'a4_a': -0.008391361373742528, 'a3_a': -0.006805831362659183,
          'delta_phi_m3': 0.36528103839369175, 'delta_phi_m4': 0.0}, {'gamma1': -0.01, 'gamma2': 0.07}]
    elif lens_index == 16:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.013202024617633541, 'e2': -0.04307405811914723, 'gamma': 2.0712479876972103, 'a4_a': 0.012309881482162046, 'a3_a': 0.013472342152438907, 'delta_phi_m3': -0.2897689043046673, 'delta_phi_m4': 0.0}, {'gamma1': 0.06, 'gamma2': 0.07}]
    elif lens_index == 17:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.12363978905818813, 'e2': 0.14055350285790214, 'gamma': 1.9485064434364303, 'a4_a': 0.0013644422519358524, 'a3_a': -0.006880462863800891, 'delta_phi_m3': -0.21502630636101866, 'delta_phi_m4': 0.0}, {'gamma1': 0.0, 'gamma2': -0.04}]
    elif lens_index == 18:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.033853490057942165, 'e2': -0.030578669991863445, 'gamma': 2.0790069719248727, 'a4_a': 0.00015188193653118965, 'a3_a': 0.00527221563870854, 'delta_phi_m3': 0.15747153771274558, 'delta_phi_m4': 0.0}, {'gamma1': 0.06, 'gamma2': -0.07}]
    elif lens_index == 19:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.040069713588407205, 'e2': -0.0860258913785972, 'gamma': 2.024470140093218, 'a4_a': 0.00810376381521321, 'a3_a': -0.003924894009763735, 'delta_phi_m3': -0.4214618266868889, 'delta_phi_m4': 0.0}, {'gamma1': 0.09, 'gamma2': 0.04}]
    elif lens_index == 20:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.2126010548992101, 'e2': 0.024915542983539554, 'gamma': 1.937935784265008, 'a4_a': 0.01502275707396579, 'a3_a': -0.004752215583806204, 'delta_phi_m3': 0.092290359073116, 'delta_phi_m4': 0.0}, {'gamma1': 0.0, 'gamma2': 0.06}]
    elif lens_index == 21:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.010969884638152535, 'e2': -0.024922939127039564, 'gamma': 1.8742539042991877, 'a4_a': -0.00969362377535908, 'a3_a': 0.005078384887289451, 'delta_phi_m3': -0.4725741997326346, 'delta_phi_m4': 0.0}, {'gamma1': -0.03, 'gamma2': 0.06}]
    elif lens_index == 22:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.021940008018599265, 'e2': 0.02396452122386181, 'gamma': 1.8755185370033827, 'a4_a': -0.006050312054248666, 'a3_a': -0.013122203824845042, 'delta_phi_m3': -0.30529941135499195, 'delta_phi_m4': 0.0}, {'gamma1': -0.05, 'gamma2': -0.07}]
    elif lens_index == 23:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': 0.2, 'e2': 0.05080821749684399, 'gamma': 2.1579395230681304, 'a4_a': -0.007234861724844146, 'a3_a': -0.004637315438040289, 'delta_phi_m3': 0.01811430160503147, 'delta_phi_m4': 0.0}, {'gamma1': -0.04, 'gamma2': 0.05}]
    elif lens_index == 24:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.08087191916720839, 'e2': -0.04086435788939368, 'gamma': 1.84514891037334, 'a4_a': -0.01228094773535638, 'a3_a': -0.0030478033716739954, 'delta_phi_m3': 0.4817289935614363, 'delta_phi_m4': 0.0}, {'gamma1': 0.015, 'gamma2': 0.05}]
    elif lens_index == 25:
        true_params = [{'theta_E': 1.0, 'center_x': 0.0, 'center_y': 0.0, 'e1': -0.03556747044943813, 'e2': -0.12566208151258698, 'gamma': 2.035094644416148, 'a4_a': 0.006042882987330165, 'a3_a': -0.008121659523875422, 'delta_phi_m3': 0.3875930895147712, 'delta_phi_m4': 0.0}, {'gamma1': 0.04, 'gamma2': 0.04}]

    phiq, q = ellipticity2phi_q(true_params[0]['e1'], true_params[0]['e2'])
    phi_gamma_ext, gamma_ext = shear_cartesian2polar(true_params[1]['gamma1'], true_params[1]['gamma2'])
    true_params = {**true_params[0], **true_params[1]}
    true_params['phi_gamma_ext'] = phi_gamma_ext
    true_params['gamma_ext'] = gamma_ext
    true_params['q'] = q
    true_params['phi_q'] = phiq
    true_params_out = {}
    for param in macromodel_samples_list:
        if param == 'a3_a_cos':
            true_params_out[param] = true_params['a3_a'] * np.cos(3 * (phiq + true_params['delta_phi_m3']))
        elif param == 'a4_a_cos':
            true_params_out[param] = true_params['a4_a'] * np.cos(4 * (phiq + true_params['delta_phi_m4']))
        elif param == 'gamma_cos_phi_gamma':
            true_params_out[param] = gamma_ext * np.cos(2 * phi_gamma_ext)
        elif param == 'q_cos_phi':
            true_params_out[param] = q * np.cos(phiq)
        else:
            true_params_out[param] = true_params[param]
    return true_params_out
