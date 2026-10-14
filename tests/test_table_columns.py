"""A table whose methods are its columns: a number's method is read from the header, by the order of the header's
labels over the row's cells (a header set a few characters off its columns still names them), and a group label over
sub-columns covers the run centred under it. The lines are real papers' tables as laid out."""

import pytest

from openlab.study.schema import column_check

KREF = """ Dataset     CRPS      Ours    Gauss               ER-S             ER-E    Ours     Gauss              ER-S              ER-E
 ETTh1       CRPS      0.436   0.466              0.456             0.453   0.379    0.411             0.402              0.398"""
KREF_HEAD = "Dataset     CRPS      Ours    Gauss               ER-S             ER-E    Ours     Gauss              ER-S"
KREF_ROW = "ETTh1       CRPS      0.436   0.466              0.456             0.453   0.379    0.411             0.402"

SCALING = """ Dataset       d                                            SCO                       SC2
                     Max          CHR         TSCP
                                                          γ=1−α         γ=0.8       γ=1−α          γrect
 ANSUR-II      2  3.40±0.01      3.33±0.01  3.34±0.01  3.35±0.01  3.34±0.01  3.34±0.01  3.35±0.01
 ENB           2  1.74±0.02      1.38±0.04  1.32±0.02  1.34±0.02  1.33±0.02 1.32±0.02   1.36±0.02"""
SCALING_HEAD = "Dataset       d                                            SCO                       SC2\nMax          CHR " \
               "        TSCP\nγ=1−α         γ=0.8       γ=1−α          γrect"
SCALING_ROW = "ENB           2  1.74±0.02      1.38±0.04  1.32±0.02  1.34±0.02  1.33±0.02 1.32±0.02   1.36±0.02"


@pytest.mark.parametrize("label,value,under", [
    ("Ours", 0.436, True), ("Gauss", 0.466, True), ("ER-E", 0.453, True), ("ER-S", 0.402, True),
    ("Gauss", 0.456, False), ("ER-S", 0.453, False)])
def test_one_header_line(label, value, under):
    assert (column_check(KREF, KREF_ROW, KREF_HEAD, label, value) == "") is under


@pytest.mark.parametrize("label,value,under", [
    ("Max", 1.74, True), ("CHR", 1.38, True), ("TSCP", 1.32, True), ("SCO / γ=1−α", 1.34, True),
    ("SC2 / γ=0.8", 1.33, True), ("SC2 / γ=1−α", 1.32, True), ("SC2 / γrect", 1.36, True),
    ("SCO / γ=1−α", 1.33, False), ("SC2 / γ=1−α", 1.34, False), ("Max", 1.38, False), ("SC2 / γ=0.8", 1.34, False)])
def test_grouped_header_levels(label, value, under):
    assert (column_check(SCALING, SCALING_ROW, SCALING_HEAD, label, value) == "") is under


def test_a_label_the_header_does_not_show_is_named():
    assert "does not show `SC3`" in column_check(SCALING, SCALING_ROW, SCALING_HEAD, "SC3 / γ=0.8", 1.33)
