classdef mountain_car_1 < rl.env.MATLABEnvironment
  properties
    open_env = py.gym.make('MountainCar-v0'); 
  end
  methods
    function this = mountain_car_1()
      ObservationInfo = rlNumericSpec([2 1]);
      ObservationInfo.Name = 'MountainCar Descreet';
      ObservationInfo.Description = 'Position, Velocity';
      ActionInfo = rlFiniteSetSpec([0 1 2]);
      ActionInfo.Name = 'Acceleration direction';
      this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
    end
    function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
      result = cell(this.open_env.step(int16(Action))); 
      Observation = double(result{1})'; 
      Reward = double(result{2});
      IsDone = double(result{3});
      LoggedSignals = [];
      if (Observation(1)>=0.4)
            Reward = 0;
            IsDone = 1;
      end
    end
    function InitialObservation = reset(this)
      result = this.open_env.reset();
      InitialObservation = double(result)'; 
    end
  end
end 