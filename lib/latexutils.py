import math
import pandas as pd
table_placeholder = '–'

def JFE_comply(latex):
    """JFE uses the old style horizontal rule, see:
    https://tex.stackexchange.com/questions/156122/booktabs-what-is-the-difference-between-toprule-and-hline
    so we have to replace the more modern horizontal rules with the older ones
    """
    latex = latex.replace('\\toprule', '\\hline\\noalign{\\smallskip}')
    latex = latex.replace('\\midrule', '\\noalign{\\smallskip}\\hline\\noalign{\\smallskip}')
    latex = latex.replace('\\bottomrule', '\\noalign{\\smallskip}\\hline')
    return latex

def full_width_table(latex):
    """Hack to make tables full page width
    """
    latex = latex.replace('\\begin{table}', '\\begin{table*}[p]')
    latex = latex.replace('\\end{table}', '\\end{table*}')
    return latex

def sideways_table(latex):
    """Hack to make tables sideways
    """
    latex = latex.replace(r'\begin{table*}[p]', r'\begin{sidewaystable*}[p]')
    latex = latex.replace(r'\end{table*}', r'\end{sidewaystable*}')
    
    # Fix column spacing so columns are evenly spaced across the page
    latex = latex.replace(r'\begin{tabular}{', r'\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}')
    latex = latex.replace(r'\end{tabular}', r'\end{tabular*}')

    return latex

def include_table_footnote(latex, footnote):
    """Hack to include custom table footnote after "\end{tabular}"
    """
    # Regular tables
    latex = latex.replace(r'\end{tabular}',
                          r"""\end{tabular}
                              \raggedright
                              \footnotesize{%s}""" % footnote)
    
    # Full width sideways tables
    latex = latex.replace(r'\end{tabular*}',
                          r"""\end{tabular*}
                              \raggedright
                              \footnotesize{%s}""" % footnote)
    return latex

def display_Q(Q):
    if Q == float('inf'):
        return r'$\infty$'
    else:
        return str(int(Q))

def siunitx_num(s):
    """Format string s to return 
    '\num{s}'
    so the siunitx latex package correctly display scientific notation
    """
    if s == table_placeholder or pd.isnull(s):
        return table_placeholder
    else:
        return r'\num{'+ str(s) +'}'
    
def cite(bibtex_strings):
    citation = f"\onlinecite{{{','.join(bibtex_strings)}}}"
    return citation

def CustomLogarithmicFormatter(x, pos=None):
    """This formatter is intended for non-scientific notation tick labels
    on logarithmic axes. It avoids displaying trailing zeros by adjusting
    the rounding for values less than 1.
    """
    if x < 1:
        round_places = int(-1 * math.log10(x))
        x_formatted = format(x, f'0.{round_places}f')
    else:
        x_formatted = format(x, '0.0f')
    return x_formatted

def latex_table_to_csv(df, output_path, header_map=None):
    """Convert a DataFrame formatted for LaTeX tables to a clean CSV file.
    
    Args:
        df (pandas.DataFrame): DataFrame that would normally be passed to to_latex()
        output_path (str): Path where the CSV file should be saved
        header_map (dict, optional): Dictionary mapping current column names to desired CSV headers.
                                   If None, uses current column names.
    
    This function cleans LaTeX formatting from the DataFrame before saving to CSV:
    - Removes \num{} wrappers from numbers
    - Replaces LaTeX placeholders with empty strings
    - Removes LaTeX formatting from column headers
    - Handles citations appropriately
    - Converts scientific notation to plain numbers
    """
    # Create a copy to avoid modifying the original DataFrame
    df_clean = df.copy()
    
    def clean_cell(val):
        if pd.isnull(val):
            return ''
        val = str(val)
        
        # Remove \num{} wrapper
        if r'\num{' in val:
            val = val.replace(r'\num{', '').replace('}', '')
            
        # Remove LaTeX citations
        if r'\onlinecite{' in val:
            val = val.replace(r'\onlinecite{', '').replace('}', '')
            
        # Replace table placeholder with empty string
        if val == table_placeholder:
            return ''
            
        # Remove other LaTeX formatting
        val = val.replace('$', '').replace(r'\rm', '').replace(r'\mathrm', '')
        val = val.replace('{', '').replace('}', '')
        val = val.replace(r'\si{', '').replace(r'\thead{', '')
        val = val.replace(r'\\', '')
        
        return val.strip()
    
    # Apply cleaning to all elements
    df_clean = df_clean.applymap(clean_cell)
    
    # Rename columns if header map provided
    if header_map is not None:
        df_clean = df_clean.rename(columns=header_map)
    
    # Save to CSV
    df_clean.to_csv(output_path, index=False)

    