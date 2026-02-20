from lab1.tools.get_data_column import get_data_column
from lab1.tools.custom_describe import analyze

analyze("Number of institutes(thsd)", get_data_column(1))
analyze("Number of places(thsd)", get_data_column(2))
analyze("Number of persons(thsd)", get_data_column(3))
analyze("Coverage of children(%)", get_data_column(4))


