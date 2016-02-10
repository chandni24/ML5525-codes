function [a] = takeStep(i1, i2)

    global X Y E K eps C L_alpha objF z;

    a = 0;
    %If the two points chosen are same, we don't optimize using these
    if (X(i1,:) == X(i2,:))
        return;
    end

    %initialize temp variables
    alph1 = L_alpha(i1);
    alph2 = L_alpha(i2);

    y1 = Y(i1);
    y2 = Y(i2);
    E1 = E(i1);
    E2 = E(i2);
    s = y1*y2;

    %Compute L, H
    if y1==y2%points belong to the same class
        L = max(0, (alph1 + alph2 - C)); %assumed that the values in L_alpha are old
        H = min(C, (alph1 + alph2));

    else if y1~=y2%points belong to different classes
            L = max(0, (alph2 - alph1)); %assumed that the values in L_alpha are old
            H = min(C, (C + alph2 - alph1));
        end
    end

    %If the lower and higher bounds are same, we don't use the current two
    %points for optimization
    if L == H
        return;
    end

    eta = 2*K(i1,i2) - K(i1,i1) + K(i2,i2);
    %eta

    %usual case
    if (eta < 0)
        a2 = alph2 - y2*(E1-E2)/eta;
        %a2
        if (a2 < L)
            a2 = L;
        elseif (a2 > H)
            a2 = H;
        end
    else
        % calculate obj functions since objective function is negative
        s = y1* y2;
        f1 = y1 * (E1) - alph1 * K(i1,i1) - s* alph2 * K(i1,i2);
        f2 = y2 * (E2) - s *alph1 * K(i1,i2) - alph2 * K(i2,i2);
        L1 = alph1 + s * (alph2 - L);
        H1 = alph1 + s * (alph2 - H);
        objL = L1*f1 + L*f2 + 0.5 * L1^2 * K(i1,i1) + 0.5 * L^2 * K(i2,i2) + s * L * L1 * K(i1,i2);
        objH = H1*f1 + H*f2 + 0.5 * H1^2 * K(i1,i1) + 0.5 * H^2 * K(i2,i2) + s * H * H1 * K(i1,i2);

        %set a2
        if (objL < objH - eps)
            a2 = L;
        elseif(objL > objH + eps)
            a2 = H;
        else
            a2 = alph2;
        end
    end

    %check to see if update was made between a2 and alph2, if not, return
    if (abs(a2-alph2) < eps*(a2+alph2+eps))
        return;
    end

    a1 = alph1 + s*(alph2 - a2);

    %Update threshold to reflect change in Lagrange multipliers - not
    %considered here

    %Update weight vector to reflect change in a1 & a2, if linear SVM - not
    %needed here

    %Update error cache using new Lagrange multipliers
    E(i2) = SVM(i2) - y2; %(check in error cache)

    %Store a1 in the alpha array
    L_alpha(i1) = a1;
    %Store a2 in the alpha array
    L_alpha(i2) = a2;
    objF(z) = objFnc_SMO;
    z = z+1;

    a = 1;
    return;