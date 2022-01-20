function Hd = DEM(RD,labeldistrube)
   
   a1 =sort(labeldistrube);
   numv = hist(labeldistrube, 75);
   hists_cumsum = cumsum(numv);
   level = 1;
   const_a = level / length(a1);
   hists_cdf = const_a * hists_cumsum;
   RD=uint16(RD);
   RD(RD>7499)=7499;
   Hd = hists_cdf(RD+1);
end

