#%%
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import inspect

time = np.array(['2002-01-01T01:35:30.150Z','2002-01-02T06:23:30.143Z','2002-01-03T20:43:30.137Z','2002-01-05T11:11:30.133Z','2002-01-06T20:27:30.132Z','2002-01-07T23:59:30.131Z','2002-01-09T06:23:30.130Z','2002-01-10T11:11:30.130Z','2002-01-11T19:11:30.129Z','2002-01-12T23:59:30.128Z','2002-01-14T06:23:30.127Z','2002-01-15T11:11:30.125Z','2002-01-16T17:35:30.123Z'])
c_x = np.array([5.124912719727E+02,5.126451416016E+02,5.129639485677E+02,5.128095703125E+02,5.129006856283E+02,5.128635253906E+02,5.127472534180E+02,5.129226379395E+02,5.131047363281E+02,5.130963745117E+02,5.128045349121E+02,5.128495178223E+02,5.130968627930E+02])
c_y = np.array([5.121821899414E+02,5.121716308594E+02,5.121700744629E+02,5.121723937988E+02,5.121798197428E+02,5.121746215820E+02,5.121821594238E+02,5.121891784668E+02,5.122049255371E+02,5.122990417480E+02,5.123047180176E+02,5.122811279297E+02,5.122782287598E+02])
r_sun = np.array([4.976492972133E+02,4.976459074648E+02,4.976371283928E+02,4.976233208255E+02,4.976072535491E+02,4.975910001418E+02,4.975698944716E+02,4.975467581566E+02,4.975174082688E+02,4.974876625971E+02,4.974528064124E+02,4.974165000041E+02,4.973747211434E+02])

sp1 = np.array([[739.5,647.5],[857.0,642.0],[959.0,630.0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
sp2 = np.array([[673.0,500.0],[799.0,498.0],[927.0,490.0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
sp3 = np.array([[236.0,579.0],[359.0,582.0],[537.0,586.0],[713.0,586.0],[849.0,580.0],[935.0,572.0],[993.0,562.0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]])
sp4 = np.array([[117.5,360.5],[207.0,364.0],[361.0,368.0],[529.0,374.0],[675.0,376.0],[783.0,372.0],[879.0,366.0],[945.0,360.0],[0,0],[0,0],[0,0],[0,0],[0,0]])
sp5 = np.array([[0,0],[0,0],[0,0],[125.0,644.0],[237.0,652.0],[351.0,656.0],[471.0,666.0],[603.0,668.0],[743.0,664.0],[847.0,660.0],[929.0,652.0],[0,0],[0,0]])
sp6 = np.array([[0,0],[0,0],[0,0],[111.0,394.0],[211.0,402.0],[319.0,406.0],[451.0,406.0],[581.0,404.0],[719.0,400.0],[827.0,390.0],[917.0,382.0],[0,0],[0,0]])
sp7 = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[139.0,768.0],[211.0,776.0],[315.0,780.0],[423.0,784.0],[545.0,784.0],[657.0,784.0],[767.0,778.0]])
sp8 = np.array([[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[97.00,590.0],[191.0,598.0],[307.0,606.0],[447.0,610.0]])

def cal(sp,c_x,c_y,r_sun):
    latitude = np.zeros(len(time))
    longitude = np.zeros(len(time))

    for i in range(len(time)):
        latitude[i] = np.arcsin( (sp[i][1] - c_y[i]) / (r_sun[i]) )
        longitude[i] = np.arcsin( (sp[i][0] - c_x[i]) / (r_sun[i]) )

    return latitude, longitude

SP1 = cal(sp1,c_x,c_y,r_sun)
SP2 = cal(sp2,c_x,c_y,r_sun)
SP3 = cal(sp3,c_x,c_y,r_sun)
SP4 = cal(sp4,c_x,c_y,r_sun)
SP5 = cal(sp5,c_x,c_y,r_sun)
SP6 = cal(sp6,c_x,c_y,r_sun)
SP7 = cal(sp7,c_x,c_y,r_sun)
SP8 = cal(sp8,c_x,c_y,r_sun)

time_date = []
time_second = []
time_day = []
time_dateo = []

for i in range(len(time)):
    time_date.append(datetime.fromisoformat(time[i][:-1] + '+00:00'))

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def linfit(x, y):
    p_coeff, residuals, _, _, _ = np.polyfit(x, y, 1, full=True)
    n = len(x)
    D = sum(x**2) - 1./n * sum(x)**2
    x_bar = np.mean(x)
    dm_squared = 1./(n-2)*residuals[0]/D
    dc_squared = 1./(n-2)*(D/n + x_bar**2)*residuals[0]/D
    dm = np.sqrt(dm_squared)
    dc = np.sqrt(dc_squared)
    return p_coeff[0], dm, p_coeff[1], dc

for i in range(len(time)):
    time_second.append(time_date[i].timestamp() - time_date[0].timestamp())
    time_day.append((time_date[i].timestamp() - time_date[0].timestamp())/86400)
    time_dateo.append(time_date[i].strftime('%m-%d'))

for a in [SP1,SP2,SP3,SP4,SP5,SP6,SP7,SP8]:
    #long = np.rad2deg(a[1][~np.isnan(a[1])])
    plt.plot(time_date,a[1],'.-',label=f'{retrieve_name(a)[0]}')
    plt.xticks(rotation=90)
    #plt.ylim(-1.5,1.5)
    plt.xlabel('Time UTC')
    plt.ylabel('Longitude / deg')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

plt.savefig("UTCtime.pdf")

fig = plt.figure()
ax1 = fig.add_subplot(111)

for a in [SP1,SP2,SP3,SP4,SP5,SP6,SP7,SP8]:
    long = np.rad2deg(a[1][~np.isnan(a[1])])
    time_day_a = np.asarray(time_day, dtype=np.float32)
    time_day1 = time_day_a[~np.isnan(a[1])]
    ax1.plot(time_day1,long,'.-',label=f'{retrieve_name(a)[0]}')
    print(type(retrieve_name(a)[0]))
    if retrieve_name(a)[0] != 'SP6':
        ax1.annotate(f"{retrieve_name(a)[0]}", (time_day1[-1]+0.25,long[-1]-2))
    if retrieve_name(a)[0] == 'SP6':
        ax1.annotate(f"{retrieve_name(a)[0]}", (time_day1[-1]+0.25,long[-1]-10))
    ax1.set_xlim(-1,17)
    ax1.set_xlabel('Time / days')
    ax1.set_ylabel('Longitude / deg')
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')



ax2 = ax1.twiny()
ax2.set_xlabel('Time UTC')
ax2.set_xlim(-1,17)
ax2.set_xticks(range(0,17,2))
ax2.set_xticklabels(['01-01','01-03','01-05','01-07','01-09','01-11','01-13','01-15','01-17'])
#ax2.xaxis.set_tick_params(rotation=80)
plt.tight_layout()
plt.savefig("daytime.pdf")

fig2 = plt.figure()
ax3 = fig2.add_subplot(111)

for i in [SP1,SP2,SP3,SP4,SP5,SP6,SP7,SP8]:
    x = i[1]
    x1 = x[~np.isnan(x)]
    time_day_a = np.asarray(time_day, dtype=np.float32)
    time_day1 = time_day_a[~np.isnan(x)]
    slope, intercept = np.polyfit(time_day1, x1, 1)
    m, dm, c, dc = linfit(time_day1, x1)
    #print(f'The slope of {retrieve_name(i)[0]} = {m} +- {dm}')
    period = (2*np.pi)/m
    s_period = (period * 365.25)/(period+365.25)
    lat = np.rad2deg(i[0][~np.isnan(x)])
    av_lat = np.average(lat)
    botlat = av_lat - min(lat)
    toplat = max(lat) - av_lat
    lat_range = np.array([botlat,toplat]).reshape((2,1))
    #print(f"Period:{period}, SPeriod:{s_period}, Average latitude:{av_lat}")
    ax3.errorbar(s_period,av_lat,yerr=lat_range,fmt='.',label=f'{retrieve_name(i)[0]}')
    if retrieve_name(i)[0] == 'SP3':
        ax3.annotate(f"{retrieve_name(i)[0]}", (s_period+0.1,av_lat-2))
    if retrieve_name(i)[0] != 'SP3':
        ax3.annotate(f"{retrieve_name(i)[0]}", (s_period+0.1,av_lat-1))
    ax3.set_xlabel('Sidereal period / days')
    ax3.set_ylabel('Avergae latitude / deg')
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    print(f"{retrieve_name(i)[0]} & ${np.rad2deg(m) : .1f} \pm {np.rad2deg(dm) + np.rad2deg(m)*0.05 : .1f}$ & ${period : .1f} \pm {(dm + m*0.05)/m * period : .1f}$ \u005c\u005c")

x = np.linspace(25,31,100)
y1 = np.sqrt(150*x - 3750) +2
y2 = -np.sqrt(150*x - 3750) +2
ax3.plot(x,y1,'k--',alpha=0.4)
ax3.plot(x,y2,'k--',alpha=0.4)
plt.tight_layout()
plt.savefig("period.pdf")
# %%
for i in [SP1,SP2,SP3,SP4,SP5,SP6,SP7,SP8]:
    SPi = np.asarray(i, dtype=np.float32)
    SPi[np.isnan(SPi)] = 0
    if retrieve_name(i)[0] == "SP1":
        SP1 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP2":
        SP2 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP3":
        SP3 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP4":
        SP4 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP5":
        SP5 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP6":
        SP6 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP7":
        SP7 = np.rad2deg(SPi)
        SPi = None
    if retrieve_name(i)[0] == "SP8":
        SP8 = np.rad2deg(SPi)
        SPi = None
for i in [SP1,SP2,SP3,SP4,SP5,SP6,SP7,SP8]:
    print(np.shape(i))

for i in range(len(time)):
    print(f"{time_date[i].strftime('%Y-%m-%d %H:%M:%S')} & {SP1[0][i] : .1f} & {SP1[1][i] : .1f} & {SP2[0][i] : .1f} & {SP2[1][i] : .1f} & {SP3[0][i] : .1f} & {SP3[1][i] : .1f} & {SP4[0][i] : .1f} & {SP4[1][i] : .1f} & {SP5[0][i] : .1f} & {SP5[1][i] : .1f} & {SP6[0][i] : .1f} & {SP6[1][i] : .1f} & {SP7[0][i] : .1f} & {SP7[1][i] : .1f} & {SP8[0][i] : .1f} & {SP8[1][i] : .1f} ")
# %%
for i in [SP1,SP2,SP3,SP4,SP5,SP6,SP7,SP8]:
    x = i[1]
    x1 = x[~np.isnan(x)]
    time_day_a = np.asarray(time_day, dtype=np.float32)
    time_day1 = time_day_a[~np.isnan(x)]
    slope, intercept = np.polyfit(time_day1, x1, 1)
    m, dm, c, dc = linfit(time_day1, x1)
    #m = uc.ufloat(m,(dm + 0.05*m))
    period = (2*np.pi)/m
    s_period = (period * 365.25)/(period+365.25)
    lat = np.rad2deg(i[0][~np.isnan(x)])
    av_lat = np.average(lat)
    botlat = av_lat - min(lat)
    toplat = max(lat) - av_lat
    lat_range = np.array([botlat,toplat]).reshape((2,1))
    print(f'{retrieve_name(i)[0]} & ${av_lat : .1f} ^\u007b{max(lat)-av_lat : .1f}\u007d _\u007b{min(lat)-av_lat : .1f}\u007d $ & ${s_period : .1f} \pm {(dm + m*0.05)/m * period : .1f}$ \u005c\u005c')

# %%
import uncertainties as uc
# %%
for i in range(len(time)):
    print(f"{time_date[i].strftime('%Y-%m-%d %H:%M:%S')} & {c_x[i] : .1f} & {c_y[i] : .1f} & {r_sun[i] : .1f} \u005c\u005c")
# %%
from astropy import constants as const
#L = I*w
# I = (2/5)M(R^2)
#w = 2*pi/T
P = uc.ufloat(25.0,2.0)
m = const.M_sun
r = const.R_sun
I = (2/5)*m*(r**2)
w = (2*np.pi)/(P*24*60*60)
L = I * w
print(f"{L : .3g}")
# %%
