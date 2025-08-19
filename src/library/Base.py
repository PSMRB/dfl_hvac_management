"""
Gather the functions used in classes and other functions, including Common.py
"""


def getcolumnname(subcolumn_name: str, columns: list) -> str:
    """
    Function to return the full name of the column given only one part of the name (subcolumn_name)
    :param subcolumn_name: the known part of the column name
    :param columns: the list containing the name of all the columns
    :return: the list of all column names which contain the subcolumn_name. If there is only one element, return that
    element (not contained in a list)
    """
    all_matches = [name for name in columns if subcolumn_name in name]
    if len(all_matches) == 1:
        return all_matches[0]
    else:
        return all_matches


