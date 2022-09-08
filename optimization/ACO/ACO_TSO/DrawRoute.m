function DrawRoute(C, R)
N = length(R);
scatter(C(:, 1), C(:, 2));
hold on
plot([C(R(1), 1), C(R(N), 1)], [C(R(1), 2), C(R(N), 2)], 'g');
for ii = 2: N
    hold on
    plot([C(R(ii - 1), 1), C(R(ii), 1)], [C(R(ii - 1), 2), C(R(ii), 2)], 'g');
    pause(0.1)
    frame = getframe(gcf);
    imind = frame2im(frame);
    [imind, cm] = rgb2ind(imind, 256);
    if ii == 2
        imwrite(imind, cm, 'test.gif', 'gif', 'Loopcount', inf, 'DelayTime', 1e-4);
    else
        imwrite(imind, cm, 'test.gif', 'gif', 'WriteMode', 'append', 'DelayTime', 1e-4);
    end
end
title('旅行商规划');