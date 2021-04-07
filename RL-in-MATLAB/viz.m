function viz(env,net,numep)
for i=1:numep
    state=single(env.open_env.reset());
    done = 0;
    count_step = 0;
    while(~done && count_step<200)
    count_step = count_step +1;
    [~,action] = max(net.predict(state));
    out = cell(env.open_env.step(int16(action-1)));
    env.open_env.render();    
    n_state = single(out{1});
    done = single(out{3});
    if (n_state(1)>=0.49)
        done=1;
    end
    state=n_state;
    end
end
env.open_env.close();
end
