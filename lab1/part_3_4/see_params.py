from lab1.tools.get_data_column import get_data_column
from lab1.tools.custom_describe import c_describe

c_describe("Number of institutes(thsd)", get_data_column(1))
c_describe("Number of places(thsd)", get_data_column(2))
c_describe("Number of persons(thsd)", get_data_column(3))
c_describe("Coverage of children(%)", get_data_column(4))


