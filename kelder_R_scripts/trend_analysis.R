# load the data
# dir='//home/timok/timok/SALIENSEAS/SEAS5/ensex'
# plotdir=paste0(dir,'/statistics/multiday/plots')
# dir='/home/timok/Documents/ensex'
# plotdir='/home/timok/Documents/ensex/R/graphs'
# dir='C:/Users/Timo/Documents/GitHub/EnsEx/Data'
# plotdir='/home/timok/Documents/ensex/R/graphs'

dir='C:/Users/gytk3/OneDrive - Loughborough University/GitHub/EnsEx/Data'
source('Load_data.R')
library(extRemes)
library("ggpubr")

# Columns of Member, Leadtime, Year
names(dimnames(Extremes_WC)) <- c('Member', 'Leadtime', 'Year')
names(dimnames(Extremes_SV)) <- c('Member', 'Leadtime', 'Year')
df_WC=adply(Extremes_WC, 1:3)
df_SV=adply(Extremes_SV, 1:3)
obs=Extremes_obs[as.character(1981:2015)]

year_vector=as.numeric(levels(df_WC$Year))[df_WC$Year] ###The year is a factor, extract the values  

# Compare the 1981 and 2015 return value plots
# A GEV distribution including parameters that linearly relate to the time period 1981-2015 is fitted to the UNSEEN ensemble. The resulting return value plots for the covariates 1981 and 2015 are shown here.
extremes_wc= df_WC$V1 * mean(obs)/mean(df_WC$V1) ## we create a mean bias corrected series  

rperiods = c(2, 5, 10, 20, 50, 80, 100, 120, 200, 250, 300, 500, 800,2000,5000)

RV_ci <- function(extremes,covariate,return_period,covariate_values,GEV_type) { ## A function to fit the GEV and obtain the return values 
  fit <- fevd(extremes, type = GEV_type, location.fun = ~ covariate, ##Fitting the gev with a location and scale parameter linearly correlated to the covariate (years)
               scale.fun = ~ covariate, use.phi = TRUE)

  params_matrix <- make.qcov(fit, vals = list(mu1 = covariate_values,phi1 = covariate_values)) #Create a parameter matrix for the GEV fit
  rvs=ci.fevd(fit,alpha = 0.05,type='return.level',return.period = return_period,method ="normal",qcov=params_matrix)  #Calculate the return values and confidence intervals for each year   
  return(rvs)
}

Plot_non_stationary <- function(GEV_type) {
  
rvs_wc_1981=RV_ci(extremes = extremes_wc,covariate = c(df_WC$Year),return_period = rperiods,covariate_values = 1,GEV_type = GEV_type) ##calc the return values
colnames(rvs_wc_1981) = c('S5_1981_l','S5_1981','S5_1981_h','S5_1981_sd') #Rename the column

rvs_wc_2015=RV_ci(extremes = extremes_wc,covariate = c(df_WC$Year),return_period = rperiods,covariate_values = 35,GEV_type = GEV_type)
colnames(rvs_wc_2015) = c('S5_2015_l','S5_2015','S5_2015_h','S5_2015_sd')

rvs_obs_1981=RV_ci(extremes = obs,covariate = c(1:35),return_period = rperiods,covariate_values = 1,GEV_type = GEV_type)
colnames(rvs_obs_1981) = c('Obs_1981_l','Obs_1981','Obs_1981_h','Obs_1981_sd') #Rename the col

rvs_obs_2015=RV_ci(extremes = obs,covariate = c(1:35),return_period = rperiods,covariate_values = 35,GEV_type = GEV_type)
colnames(rvs_obs_2015) = c('Obs_2015_l','Obs_2015','Obs_2015_h','Obs_2015_sd')

rvs_WC=data.frame(cbind(rvs_wc_1981,rvs_wc_2015,rvs_obs_1981,rvs_obs_2015,rperiods))

# cols=c("S5_1981"="#f04546","S5_2015"="#3591d1","Obs_1981"="#62c76b","Obs_2015"="#62c76b")
p_wc=ggplot(data = rvs_WC,aes(x=rperiods))+
  geom_line(aes(y = S5_1981),col='black')+
  geom_ribbon(aes(ymin=S5_1981_l,ymax=S5_1981_h),fill='black',alpha=0.5,show.legend = T)+
  geom_line(aes(y = S5_2015),col='red')+
  geom_ribbon(aes(ymin=S5_2015_l,ymax=S5_2015_h),fill='red', alpha=0.5,show.legend = T)+
  geom_line(aes(y = Obs_1981),col='black')+
  geom_ribbon(aes(ymin=Obs_1981_l,ymax=Obs_1981_h),fill='black', alpha=0.1,show.legend = T)+
  geom_line(aes(y = Obs_2015),col='red')+
  geom_ribbon(aes(ymin=Obs_2015_l,ymax=Obs_2015_h),fill='red', alpha=0.1,show.legend = T)+
  scale_x_continuous(trans='log10')+
  theme_classic()+
  xlab('Return period (years)')+
  ylab('Three-day precipitation (mm)')

rvs_sv_1981=RV_ci(extremes = df_SV$V1,covariate = c(df_WC$Year),return_period = rperiods,covariate_values = 1,GEV_type = GEV_type) ##calc the return values
colnames(rvs_sv_1981) = c('S5_1981_l','S5_1981','S5_1981_h','S5_1981_sd') #Rename the column

rvs_sv_2015=RV_ci(extremes = df_SV$V1,covariate = c(df_WC$Year),return_period = rperiods,covariate_values = 35,GEV_type = GEV_type)
colnames(rvs_sv_2015) = c('S5_2015_l','S5_2015','S5_2015_h','S5_2015_sd')

rvs_SV=data.frame(cbind(rvs_sv_1981,rvs_sv_2015,rperiods))


cols=c("1981"="black","2015"="red")
p_sv=ggplot(data = rvs_SV,aes(x=rperiods))+
  geom_line(aes(y = S5_1981),col='black')+
  geom_ribbon(aes(ymin=S5_1981_l,ymax=S5_1981_h,fill="1981"),alpha=0.5)+
  geom_line(aes(y = S5_2015,colour="2015"),col='red')+
  geom_ribbon(aes(ymin=S5_2015_l,ymax=S5_2015_h,fill="2015"), alpha=0.5)+
  scale_x_continuous(trans='log10')+
  theme_classic()+
  scale_fill_manual(name="Years",values=cols) +
  theme(axis.title.y=element_blank())+
  xlab('Return period (years)')+
  ylab('Three-day precipitation (mm)')

ggarrange(p_wc, p_sv,
          labels = c("c", "d"),
          legend='top',
          common.legend = T,
          hjust = c(-0.5,1),
          ncol = 2, nrow = 1)
}

CD=Plot_non_stationary(GEV_type = 'GEV')
CD