
NAME:  AAUP Faculty Salary data
TYPE:  Census (mostly)
SIZE:  1161 colleges, 16 variables


DESCRIPTIVE ABSTRACT:
Data are from the American Association of University Professors
(AAUP) annual faculty salary survey of American colleges and
universities.  They include average salary and overall
compensation, broken down by full, associate, and assistant
professor ranks.  The dataset is used for the 1995 Data Analysis
Exposition, sponsored by the Statistical Graphics Section of the
American Statistical Association.  See the file
colleges.txt for more information on the Exposition.


SOURCE:
This dataset is taken from the March-April 1994 issue of Academe.
More detailed descriptions of the variables can be found in that
source.  Thanks to Maryse Eymonerie, Consultant to AAUP, for
assistance in supplying the data.


Note: The JSE dataset archives contain two versions of this
dataset.

AAUP.DAT contains the data in a comma delimited format with a
single row of data for each college.

AAUP2.DAT contains the data in a fixed column format with three
data lines for each school and a maximum line length of 80
characters.

The format for AAUP2.DAT is described below, although the same
variables in the order given below are found on each row of
AAUP.DAT.

VARIABLE DESCRIPTIONS (AAUP2.DAT)
Fixed column format with two data lines per school

Line #1
  1 -  5   FICE (Federal ID number)
  7 - 37   College name
 38 - 39   State (postal code)
 40 - 43   Type  (I, IIA, or IIB)
 44 - 48   Average salary - full professors
 49 - 52   Average salary - associate professors
 53 - 56   Average salary - assistant professors
 57 - 60   Average salary - all ranks
 61 - 65   Average compensation - full professors
 66 - 69   Average compensation - associate professors
 70 - 73   Average compensation - assistant professors
 74 - 78   Average compensation - all ranks

Line #2
  1 -  4   Number of full professors
  5 -  8   Number of associate professors
  9 - 12   Number of assistant professors
 13 - 16   Number of instructors
 17 - 21   Number of faculty - all ranks

ALL SALARY AND COMPENSATION FIGURES ARE YEARLY IN $100'S.
Missing values are denoted with *.


STORY BEHIND THE DATA:
This dataset is used along with another (see usnews.txt) as the 
basis for the 1995 Data Analysis Exposition.  This is a special 
session at the Joint Statistical Meetings which uses a common 
dataset as a vehicle for demonstrating innovative approaches to 
analyzing and displaying data.  Salary data are for the 1993-94 
school year.


PEDAGOGICAL NOTES:
The purpose of the Data Analysis Exposition is to provide a
common dataset for individuals and groups to use to demonstrate
approaches to analyzing data and displaying statistical results.
In keeping with the spirit of this Exposition, one might ask
students to prepare posters displaying the results of their own
analyses.  The dataset contains a wealth of information which is
quite naturally of considerable interest to college students (and
faculty!).   For example, how does your school fare in the data?
Is your salary structure in line with what other characteristics
of your school suggest it should be?

SUBMITTED BY:
Robin Lock
Mathematics Department
St. Lawrence University
Canton, NY  13617
rlock@vm.stlawu.edu