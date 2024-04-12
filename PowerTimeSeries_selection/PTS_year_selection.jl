using CSV
using DataFrames
using StatsPlots
using Plots
using GLM
using StatsBase 
using Dates
using XLSX
using PlotlySave
using Distributions


# LOADING DATA FROM PTS
RAW = DataFrame(CSV.File("C:/Users/Guillermo Rilova/OneDrive - Greengo Energy/Documents/Wind/DEVELOPMENT/Wind Projects/Merida, Spain/Output/Vestas_162_6_2_Salamanca.csv",header=3, skipto=5))

# Filtering to Park PTS
PTS = RAW[:,[:"Time stamp",:"Reduced wind speed",:"Power"]]
rename!(PTS,[:"Time stamp",:"Reduced wind speed"] .=> [:date, :WS]) #Changing column name 
PTS.date = DateTime.(PTS.date,"mm/dd/yyyy HH:MM") # Converting to datetime format

# PLOT SCATTER
fc1 = RGBA(90/255,132/255,119/255,0.3)
fc2 = RGBA(224/255,187/255,33/255,1)
fc3 = RGBA(166/255,214/255,204/255,1)

# PLOT WIND SPEED
plot1 = Plots.scatter(PTS.date,PTS.WS,mode = "markers",ms=2,markerstrokewidth= 0,color=fc1, labels = "Height 59 m")
# PLOT PTS
plot2= Plots.scatter(PTS.date,PTS."Power",mode = "markers",ms=2,markerstrokewidth= 0,color=fc2, labels = "Height 59 m")
# WS VS PTS
plot3= Plots.scatter(PTS.WS,PTS."Power",mode = "markers",ms=2,markerstrokewidth= 0,color=fc3, labels = "Height 59 m")

# FUNCTIONS FOR EXTRACTING WEIBULL LATER
function weib(vect, opt=1)    
    weib_tmp = fit(Weibull,vect)
    if opt==1
        return params(weib_tmp)[1]
    else
        return params(weib_tmp)[2]
    end    
end

function weib_k(vect)
    return weib(vect)
end
function weib_A(vect)
    return weib(vect, 2)
end

#GROUPING DATA BY YEAR
PTS.year = Dates.year.(PTS.date)    
PTS_year = groupby(PTS,:year)

# Calculate metrics
PTS_stats_year = combine(PTS_year, [:WS, :WS, :WS, :WS, :Power, :Power]  .=> [mean, std, weib_k, weib_A, mean, std]) # By year
PTS_total = combine(PTS, [:WS, :WS, :WS, :WS, :Power, :Power]  .=> [mean, std, weib_k, weib_A, mean, std]) # In total

# Repeat the total value to efficiently extract the differences for each year
rep_PTS_total = repeat(PTS_total, inner=nrow(PTS_stats_year))

# Get the difference with the metrics of the full dataset
Diff_year = PTS_stats_year[:,Not(:year)] .- rep_PTS_total
Diff_abs = broadcast(abs,Diff_year)

Diff_abs_ranking = combine(Diff_abs, [:WS_mean, :WS_std, :WS_weib_k, :WS_weib_A, :Power_mean, :Power_std] .=> [sortperm,sortperm,sortperm,sortperm,sortperm,sortperm]) 
Ranking_point = Diff_abs_ranking
Ranking_WS = (Ranking_point.WS_mean_sortperm + Ranking_point.WS_std_sortperm +Ranking_point.WS_weib_k_sortperm + Ranking_point.WS_weib_A_sortperm)
Ranking_P = (Ranking_point.WS_mean_sortperm + Ranking_point.WS_std_sortperm +Ranking_point.WS_weib_k_sortperm + Ranking_point.WS_weib_A_sortperm + Ranking_point.Power_mean_sortperm + Ranking_point.Power_std_sortperm)

Ranking_TOT = (Ranking_point.Power_mean_sortperm + Ranking_point.Power_std_sortperm)


RANKING =[ PTS_stats_year.year[sortperm(Ranking_WS)] PTS_stats_year.year[sortperm(Ranking_P)] ]


# PLOTTING THE WINNER

fig1 = plot(Weibull(PTS_stats_year[PTS_stats_year.year.==RANKING[1,1],:WS_weib_k][1],PTS_stats_year[PTS_stats_year.year.==RANKING[1,1],:WS_weib_A][1]),ms=5,labels = "Best ranked", xlabel="Wind Speed m/s", title="Weibull distribution ranked by WS")
plot!(Weibull(PTS_stats_year[PTS_stats_year.year.==RANKING[size(RANKING)[1],1],:WS_weib_k][1],PTS_stats_year[PTS_stats_year.year.==RANKING[size(RANKING)[1],1],:WS_weib_A][1]),ms=5,labels = "Worst ranked", xlabel="Wind Speed m/s")
plot!(Weibull(PTS_total.WS_weib_k[1],PTS_total.WS_weib_A[1]),ms=5,labels = "All dataset", xlabel="Wind Speed m/s")


fig1 = plot(Weibull(PTS_stats_year[PTS_stats_year.year.==RANKING[1,2],:WS_weib_k][1],PTS_stats_year[PTS_stats_year.year.==RANKING[1,2],:WS_weib_A][1]),ms=5,labels = "Best ranked", xlabel="Wind Speed m/s", title="Weibull distribution ranked by Power")
plot!(Weibull(PTS_stats_year[PTS_stats_year.year.==RANKING[size(RANKING)[1],2],:WS_weib_k][1],PTS_stats_year[PTS_stats_year.year.==RANKING[size(RANKING)[1],2],:WS_weib_A][1]),ms=5,labels = "Worst ranked", xlabel="Wind Speed m/s")
plot!(Weibull(PTS_total.WS_weib_k[1],PTS_total.WS_weib_A[1]),ms=5,labels = "All dataset", xlabel="Wind Speed m/s")
