import src.settings as settings
import src.utils as utils



for year in [1995, 2015]:

	ccf_matrix = utils.load_Kbar(
		year, 'pxp', settings.DATA_DIR / ('Kbar_exio_v3_6_'+ str(year) + 'pxp.mat')
	)

	ccf_matrix.to_pickle(settings.DATA_DIR / ('ccf_matrix' + str(year) + '.pkl'))
