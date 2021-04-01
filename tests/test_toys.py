import alldecays


def test_standard_toys(data_set1):
    fit = alldecays.Fit(data_set1)
    fit.fill_toys(n_toys=2)
