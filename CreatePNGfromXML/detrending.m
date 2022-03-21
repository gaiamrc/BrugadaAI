function leads_clean = detrending(leads)
    coeff1 = modwt(leads(:,1));
    coeff1(9:13,:) = 0;
    lead1 = imodwt(coeff1);
    coeff2 = modwt(leads(:,2));
    coeff2(9:13,:) = 0;
    lead2 = imodwt(coeff2);
    coeff3 = modwt(leads(:,3));
    coeff3(9:13,:) = 0;
    lead3 = imodwt(coeff3);
    coeff4 = modwt(leads(:,4));
    coeff4(9:13,:) = 0;
    lead4 = imodwt(coeff4);
    coeff5 = modwt(leads(:,5));
    coeff5(9:13,:) = 0;
    lead5 = imodwt(coeff5);
    coeff6 = modwt(leads(:,6));
    coeff6(9:13,:) = 0;
    lead6 = imodwt(coeff6);
    coeff7 = modwt(leads(:,7));
    coeff7(9:13,:) = 0;
    lead7 = imodwt(coeff7);
    coeff8 = modwt(leads(:,8));
    coeff8(9:13,:) = 0;
    lead8 = imodwt(coeff8);
    coeff9 = modwt(leads(:,9));
    coeff9(9:13,:) = 0;
    lead9 = imodwt(coeff9);
    coeff10 = modwt(leads(:,10));
    coeff10(9:13,:) = 0;
    lead10 = imodwt(coeff10);
    coeff11 = modwt(leads(:,11));
    coeff11(9:13,:) = 0;
    lead11 = imodwt(coeff11);
    coeff12 = modwt(leads(:,12));
    coeff12(9:13,:) = 0;
    lead12 = imodwt(coeff12);
    lead1 = lead1';
    lead2 = lead2';
    lead3 = lead3';
    lead4 = lead4';
    lead5 = lead5';
    lead6 = lead6';
    lead7 = lead7';
    lead8 = lead8';
    lead9 = lead9';
    lead10 = lead10';
    lead11 = lead11';
    lead12 = lead12';
    leads_clean = [lead1 lead2 lead3 lead4 lead5 lead6 lead7 lead8 lead9 lead10 lead11 lead12];
    % figure
    % plot(lead10,'r')
    % hold on
    % plot(leads_filt(:,10),'b')
end