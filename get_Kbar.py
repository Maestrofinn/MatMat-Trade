import src.settings as settings
import src.utils as utils


years = [*range(1998, 2016)]
[years.remove(elt) for elt in [*range(2000, 2016, 5)]]

for year in years:

	ccf_matrix = utils.load_Kbar(
		year, 'pxp', settings.DATA_DIR / 'capital_consumption' / ('Kbar_'+ str(year) + 'pxp.mat')
	)

	ccf_matrix.to_pickle(settings.DATA_DIR / 'capital_consumption' / ('ccf_matrix_' + str(year) + '.pkl'))
