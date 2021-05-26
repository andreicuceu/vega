from vega.utils import find_file

composites = {'bias': r'b_{\delta,',
              'bias_eta': r'b_{\eta,',
              'beta': r'\beta_{',
              'alpha': r'\alpha_{'}


def build_names(params):
    latex_names_path = 'vega/postprocess/latex_names.txt'
    latex_names_path = find_file(latex_names_path)
    latex_full = get_latex(latex_names_path)

    latex_comp_path = 'vega/postprocess/latex_composite.txt'
    latex_comp_path = find_file(latex_comp_path)
    latex_comp = get_latex(latex_comp_path)

    latex_names = {}
    for name in params:
        if name in latex_full:
            latex_names[name] = latex_full[name]
        else:
            tracer = None
            for subset in latex_comp:
                if subset.lower() in name.lower():
                    tracer = subset

            comp_par = None
            for comp in composites:
                if comp in name:
                    comp_par = comp

            if tracer is not None and comp_par is not None:
                comp_name = comp_par + '_' + tracer
                latex_names[comp_name] = composites[comp_par] \
                    + latex_comp[tracer] + r'}'
            elif comp_par is not None:
                print('Warning: No latex name found for tracer: %s. If you'
                      ' want plots to work well, add a latex name to'
                      ' latex_composite.txt' % name[len(comp_par) + 1:])
                latex_names[name] = composites[comp_par] \
                    + name[len(comp_par) + 1:] + r'}'
            else:
                print('Warning! No latex name found for %s. Add the latex'
                      ' representation to latex_names.txt.' % name)
                latex_names[name] = name

    return latex_names


def get_latex(path):
    with open(path) as f:
        content = f.readlines()

    latex_names = {}
    for line in content:
        line = line.strip()
        if line[0] == '#':
            continue

        items = line.split()
        if len(items) > 2:
            latex = ' '.join(items[1:])
        else:
            latex = items[1]

        latex_names[items[0]] = latex

    return latex_names


def get_default_values():
    values_path = 'vega/postprocess/default_values.txt'
    with open(find_file(values_path)) as f:
        content = f.readlines()

    values = {}
    for line in content:
        line = line.strip()
        if line[0] == '#':
            continue

        items = line.split()
        values[items[0]] = {}
        values[items[0]]['limits'] = (float(items[1]), float(items[2]))
        values[items[0]]['error'] = float(items[3])

    return values
