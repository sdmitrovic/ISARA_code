import importICARTT
import numpy as np
import os
import sys
import datetime
def Merge():
	def pause():
		programPause = input("Press the <ENTER> key to continue...")


	Data_List_Str = input("Enter the list of .ict data IDs to be merged\nseparated by a comma and a space (e.g., APS, UHSAS, MICROPHYSICAL, OPTICAL): ")   
	Data_List = np.array(Data_List_Str.split(", ")).astype(str)	
	temporal_resolution = int(input("Enter the desired temporal resolution in seconds (e.g., 30): "))
	separated_data = {}
	fileID_list = {}
	for DN in Data_List:
		separated_data[DN] = {}
		IFN = [f for f in os.listdir(f'./{DN}/') if f.endswith('.ict')] 
		#IFN = IFN[0:2]
		fileID_list[DN] = np.full(len(IFN),np.nan).astype(str)
		icount = 0
		print(DN)
		for f in IFN:
			filename = f'./{DN}/{f}'
			data = importICARTT.imp(filename,1) 
			DATE = f"{data['date'][0]}{data['date'][1]}{data['date'][2]}"
			fileID_list[DN][icount] = DATE
			icount += 1
			separated_data[DN][DATE] = data


	for ID in fileID_list[Data_List[0]]:
		merged_data = {}
		for DN in Data_List:
			FOI = np.squeeze(np.where(fileID_list[DN] == ID)[0])
			avgmergdat = {}
			sepdata = separated_data[DN][fileID_list[DN][FOI]]
			Tstart = sepdata['Time_Start_Seconds'] 
			Ldata = len(Tstart)
			print(Ldata,np.nanmin(Tstart),np.nanmax(Tstart),temporal_resolution)
			tgrd = np.arange(np.nanmin(Tstart),np.nanmax(Tstart),temporal_resolution)
			for t in np.arange(len(tgrd)):
				time_idx = np.where(((Tstart>=tgrd[t]-temporal_resolution)&(Tstart<tgrd[t]+temporal_resolution)))[0]
				for key in sepdata:
					if len(sepdata[key])==Ldata:
						if key.__contains__('fmtdatetime'):
							avgdata = np.squeeze(sepdata[key][time_idx]).view('i8').mean(axis=0).astype('datetime64[s]')
						else:
							avgdata = np.nanmean(sepdata[key][time_idx])
						if key in avgmergdat:
							avgmergdat[key][t] = avgdata
						else:
							if isinstance(avgdata,datetime.datetime):
								avgmergdat[key] = np.full(len(tgrd),"NaT").astype('datetime64[s]')
							else:
								avgmergdat[key] = np.full(len(tgrd),np.nan)
							avgmergdat[key][t] = avgdata
			for key in avgmergdat:
				if key not in merged_data:
					merged_data[key] = avgmergdat[key]

		output_filename = f'pacepax-mrg{temporal_resolution}_MARINA-TOWER_{ID}_RA.npy'
		print(output_filename)
		np.save(output_filename, merged_data)  	
