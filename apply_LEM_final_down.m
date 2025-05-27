%this one interp-filter-interp by using a through-the water vertical coordinate to deal with up/down drafts 

function RESULT=apply_LEM_final_down(dives,base,uselayers,ell,Kz);
VMPnoise=1e-10;

% Input: 
%   dive number (dives, can be an array of many dives)
%   file base including the (relative) path
% e.g., base='sg_10km2B2/p005'; dives=14
% will load the file sg_10km2B2/p0050014.mat
% Output is in RESULT structure
%
% Dependence: get_seaglider_variables, nansort, nanmean, rms, NaN_interp


for I=1:length(dives);
    dive=dives(I);
disp(sprintf('dive %d \n',dive));
fname=sprintf([base,'%04d.mat'],dive);
load(fname);

%unpack Jim Bennet type dive mat files:
unpack_mat
qc_declarations
compute_dive_climb_indices

%use potential temperature:
PT_temp=theta;
%choose dive and climb indices
iwd=dive_i_corrected; %dive indices
iwp=climb_i_corrected; %climb indices

    z= ctd_depth_m(iwd);
    dens=sw_dens0(salin(iwd),PT_temp(iwd));
    rho=nansort(dens);
    idatenum = get_timestamp(dive);
    midtime=midpoint_day;
    midlat=midpoint_lat;
    midlon=midpoint_lon;

%need to be sure there is some data here:
 if length(iwd)<= 3
     dcinds=iwd;
%      NaNoutRESULTS
     continue %skip this profile b/c not long enough to filter
 end


   
   %sort the data into layers: AL, IL, BML:      
   if uselayers
    layerI= catagorize_FBClayers(PT_temp,ctd_sg_depth_m,glideangle_model,iwd,climb_i);
   else
       layerI=zeros(length(PT_temp),1);
   end
    
    %get w from pressure, note that cntrd1stderiv is a low-pass filter
    %of length 3:
    w_meas=ctr1stdiffderiv(-ctd_sg_depth_m(iwd),time(iwd));%a LPF, remember
    w= w_meas - w_model(iwd)/100;
    %OK, dude: you should NaN out points where w_model==0, the model is not
    %well behaved here and it gives spikes in w_h2o:
    ind_bad_model=find(w_model(iwd)==0);
    w(ind_bad_model)=NaN;
    
    %YOU MUST REMOVE THE TOP ~40m OF THE DIVE!!!!!!! it always looks like large
    %W, has something to do with flair:
     isurface=find(z<40);
     w(isurface)=NaN;

     
    

    
        % get the Thorpe displacements:
%in the new JSB code NaNs are in the data stream, so we need to deal with
%them in this sort algo:
iNaN=~isnan(PT_temp(iwd));
Tnonan=PT_temp(iwd(iNaN));
Znonan=z(iNaN);
    [rhosort isort]=sort(Tnonan,'descend'); %from PT_temp
    zsort=Znonan(isort);
    Ltshort=abs(Znonan - zsort);
    
     %make sure that the difference between the sorted profile and
     %the unsorted is less than the temperature error:
     terror=0.003;%error in temp sensor
     diff_is_too_small=find(abs(Tnonan-rhosort)<=terror);
     Ltshort(diff_is_too_small)=0;%these displacements could be due to noise
    
    %sow Lt into the full length vector
    Lt=NaN*z; Lt(iNaN)=Ltshort;
    

    lag_Pass1=NaN.*z;
    rmsw=NaN.*z;
    gapli=NaN.*z;


%we filter in a 'vertical distance through water' vertical coordinate so
%that reversals in depth because of strong upwelling don't cause major
%problems.

% its possible that diff(z)==0 at some point, and you can't have that for
% the interpolation, so you should just shift it by 1 cm, i.e. set the
% point where diff(z)=0 to diff(z)==.01
dz=abs(diff(z));
if any( dz==0)
    idz0=find(dz==0);
    dz(idz0)=0.01;
end

%a vertical distance through the water coordinate:
absZ=cumsum([z(1); dz]);

sampint=mean(dz);%mean separation between points

% interpolate onto a regular depth grid:
zi=[ceil(nanmin(absZ)):sampint:floor(nanmax(absZ))]';

wi_wnan=interp1(absZ,w,zi,'linear');
rhosmoothnan=interp1(absZ,rho,zi,'linear');%want to smooth the density too (for BVFQ)

%remove nan's before you filter:
wi=NaN_interp(wi_wnan);%this will NOT remove any NANs at the start or end of vector
rhosmoothi=NaN_interp(rhosmoothnan);%
%get a shorter vector with our 'bookend'NaNs:
% [wi_nonan inan]=denan(wi);
wi_nonan=denan(wi);
inonan=find(~isnan(wi));

 % set up the high-pass filter 
KzSTDY=Kz;%filter length    
FNorm= KzSTDY/(0.5*(1/sampint));
%make butterworth filter:
[b,a]=butter(4,FNorm,'high');


 %we need to be sure that the record lenght is more than 3x the filter length
 if length(wi_nonan)<= 3*length(b)
     dcinds=iwd;
     NaNoutRESULTS
     continue %skip this profile b/c not long enough to filter
 end
 
 
 %filter it:
    highpass_wi=filtfilt(b,a,wi_nonan);
%     lowpass_rho=mysmooth(rhosmoothi,20);%dens for BVFQ
    lowpass_rho=mysmooth(rhosmoothi,3);%dens for BVFQ

% interpolate back onto the original sample rate:
% hp_w=interp1(zi(inonan),highpass_wi, z,'spline');
    hp_w=interp1(zi(inonan),highpass_wi, z,'linear','extrap'); % does this make sense ?????? totally different meaning of zi and z ?
    inan=find(isnan(w));
    hp_w(inan)=NaN;
    rhosmooth=interp1(zi,lowpass_rho, z,'linear','extrap');

    %     wnlvl=0.004;%noise level for w if you use 'diff'
    wnlvl=0.002;%noise level for w if you use 'cntr1stdiffderiv'
    %probably for the fixed length, w<noise should be removed before
    %averaging:
    hp_w(hp_w<wnlvl)=0;
    
   rmsw_fixlen=[]; 
    %loop through and find rms(Lthorpe) for a fixed length scale
    for i=1:length(hp_w)
        inds=find(abs(z(i)- z)<= ell/2);
        if~isempty(inds)
            
            rmsLt= rms(Lt(inds));
            rmsw_fixlen(i)=rms(hp_w(inds));
            if rmsLt>0
                lag_Pass1(i)= rmsLt;
            else
                lag_Pass1(i)= NaN;%cant have 1/0, NaN it
            end
            
        else %no obs in this interval
            rmsw(i)=NaN;
            lag_Pass1(i)= ell;
            rmsw_fixlen(i)=NaN;
        end
        
    end
    

    %the velocity scale is abs(high-pass w)
    rmsw=abs(hp_w);

%     rmsw(rmsw<wnlvl)=NaN;
%       rmsw_fixlen(rmsw_fixlen<wnlvl)=NaN;
rmsw_fixlen(rmsw_fixlen==0)=NaN;

    
    L=lag_Pass1;
    e=((rmsw).^3)./L;%LEM esimate
    
    
        BVFQ=NaN*rhosmooth;
    for i=3:length(rhosmooth)-2
        BVFQ(i)=sqrt((9.8/1027).*((rhosmooth(i+2)-rhosmooth(i-2))./(z(i+2)-z(i-2))));
    end
    
%             BVFQ=NaN*rho;
%     for i=3:length(rho)-2
%         BVFQ(i)=sqrt((9.8/1027).*((rho(i+2)-rho(i-2))./(z(i+2)-z(i-2))));
%     end

    
    
%uncomment to make a figure of the components in the estimate for each dive
%         make_component_fig
%         figure
%         plot(rmsw,z)


    PeriodRatio=(BVFQ.*L)./((2*pi).*(rmsw./100));
IWinfluence=find(PeriodRatio>3);

   RESULT(I).dive=dive;
   RESULT(I).time=midtime;
   RESULT(I).lat=midlat;
   RESULT(I).lon=midlon;
      %the lons and lats are now not assigned for each point in the dive, you
   %need to do that for later box/bulk averaging. it's easier to do here
   %than everytime you make the depth structure
   RESULT(I).all_time=midtime*ones(size(z));
   RESULT(I).all_lat=midlat*ones(size(z));
   RESULT(I).all_lon=midlon*ones(size(z));
   
   RESULT(I).rmsw=rmsw;  
   RESULT(I).rmsw_fixlen=rmsw_fixlen(:);
    RESULT(I).dive=dive;
    RESULT(I).z=z;
    RESULT(I).w=w;
    RESULT(I).BVFQ=BVFQ;
    RESULT(I).T=PT_temp(iwd);
    RESULT(I).rho=dens;
    RESULT(I).L=L;
    RESULT(I).e=e;
    RESULT(I).PeriodRatio=PeriodRatio;
    RESULT(I).IWinfluence=IWinfluence;
    RESULT(I).hab= -(z -max(z));
    RESULT(I).layerI= layerI(iwd);
    RESULT(I).gapL=gapli;
    RESULT(I).Lt= Lt;
    
    clear dive A w z rho BVFQ IW* Per* L e
    
       [uniT, m, n] = unique(RESULT(I).T); %to interp, must have unique xvals
   %need to remove any NaN's JSB new code seems to contain NaN's sometimes:
   iNOnan=~isnan(uniT);
%     RESULT(I).z3degLevel=interp1(RESULT(I).T,RESULT(I).z,3);
    RESULT(I).z3degLevel=interp1(uniT(iNOnan),RESULT(I).z(m(iNOnan)),3);
    RESULT(I).z3=RESULT(I).z3degLevel-RESULT(I).z;
    RESULT(I).eclean=RESULT(I).e;

end

