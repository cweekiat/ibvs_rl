% import helper_functions.*
% import rl_environment.*

% Define your custom quadcopter environment
mass = 2;           % Mass of UAV
focal_length = 800; % Focal length of the camera
image_width = 640; % Width of the image plane
image_height = 480; % Height of the image plane
fov_horizontal = deg2rad(60); % 60 degrees horizontal FOV
fov_vertical = deg2rad(40);   % 40 degrees vertical FOV
dt = 0.1;

% Define state space (modify as needed)
stateLimits = [-image_width -image_height; image_width image_height]; % Image frame size
numStates = size(stateLimits, 2);

% Define action space (modify as needed)
actionLimits = [-1 -1 -1 -5; 1 1 1 15]; % Roll, Pitch, Yaw, Thrust
numActions = size(actionLimits, 2);

% Define observation space (2D XY coordinates)
numObservations = 2; % Two dimensions for X and Y coordinates
observationInfo = rlNumericSpec([numObservations 1]);
observationInfo.Name = "Camera States";
observationInfo.Description = 'U and V image coordinates';

% Define action space (roll, pitch, yaw, and thrust inputs)
numActions = 4; % Four dimensions for roll, pitch, yaw, and thrust inputs
actionInfo = rlNumericSpec([1 numActions], 'LowerLimit', actionLimits(1, :), 'UpperLimit', actionLimits(2, :));
actionInfo.Name = "Action Space";
actionInfo.Description = 'roll, pitch, yaw, and thrust inputs';

% Create reinforcement learning environment
env = rlFunctionEnv(numStates, numActions, "customStateTransition", "customIsTerminal");


% Create actor network
actorNetwork = [
    imageInputLayer([numObservations 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(numActions,'Name','output')
    tanhLayer('Name','outputScaled')
];

actorOptions = rlRepresentationOptions('LearnRate',1e-4,'GradientThreshold',1);

% Create critic network
criticNetwork = [
    imageInputLayer([numObservations 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(128,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(64,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(1,'Name','output')];

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1);

% Create actor and critic agents
actor = rlDeterministicActor(actorNetwork, observationInfo, actionInfo, actorOptions);
critic = rlValueNetwork(criticNetwork, observationInfo, criticOptions);

agent = rlACAgent(actor, critic);

% Define training options
trainingOptions = rlTrainingOptions(...
    'MaxEpisodes', 1000, ...
    'MaxStepsPerEpisode', 1000, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',200);

% Train the actor-critic agent
train(agent, env, trainingOptions);



%% ENVIRONMENT FUNCTIONS

% Define custom state transition function (modify as needed)
function [nextState, quad_state] = customStateTransition(camera_state, quad_state, action, mass, dt)
    % Implement your own quadcopter dynamics here
    % Update the state based on the given action
    % Example: nextState = myQuadcopterDynamics(state, action);
    quad_state = quadcopter_dynamics(quad_state, action, mass, dt);
    camera_position = quad_state(1:3);
    camera_orientation = quad_state(7:9);
    relative_positions = target_position - camera_position;

    R = rotationMatrix(camera_orientation);
    rotated_positions = (R * relative_positions')';

    % Check if the point is within the camera's field of view
    if isWithinFOV(rotated_positions, fov_horizontal, fov_vertical)
        % Calculate the projection using the pinhole camera model
        image_x = (focal_length / rotated_positions(1)) * rotated_positions(2);
        image_y = (focal_length / rotated_positions(1)) * rotated_positions(3);
        camera_state = [image_x image_y];
    else
        % Point is outside the field of view, set the image coordinates to NaN
        camera_state = [NaN NaN];
    end
    
    % Ensure the state stays within defined limits
    nextState = max(min(camera_state, stateLimits(2, :)), stateLimits(1, :));
end

% Define custom reward function (modify as needed)
function reward = customReward(camera_state, action, nextState)
    % Define the center of the image frame
    center_x = 0;
    center_y = 0;
    
    % Extract the current position of the point from the state
    % Assuming state contains the x and y coordinates of the point
    point_x = camera_state(1);
    point_y = camera_state(2);

    % Calculate the Euclidean distance between the point and the center
    distance_to_center = sqrt((point_x - center_x)^2 + (point_y - center_y)^2);
    
    % Define a reward scale factor
    scale_factor = 1.0; % Adjust as needed
    
    % Calculate the reward based on the distance to the center
    % Encourage the point to stay close to the center
    reward = scale_factor * (1.0 - distance_to_center / max_distance);
    
    % Optional: Penalize the agent if the point goes out of frame for a prolonged period
    if distance_to_center > max_distance
        reward = -1; % Negative reward for going out of frame
    end
end

% Define custom termination condition (modify as needed)
function isTerminal = customIsTerminal(state, episodeCount)
    % Define the maximum number of consecutive steps allowed out of frame
    max_consecutive_steps = 50; % Adjust as needed
    % Define a maximum number of training episodes
    max_episodes = 1000; % Adjust as needed

    % Extract the current position of the point from the state
    % Assuming state contains the x and y coordinates of the point
    point_x = state(1);
    point_y = state(2);
    
   % Check if the point is out of frame
    out_of_frame = ...
        (point_x < -image_width/2 || point_x > image_width/2) || ...
        (point_y < -image_height/2 || point_y > image_height/2);
    
    % If the point is out of frame, increment the counter; otherwise, reset it
    if out_of_frame
        steps_out_of_frame = steps_out_of_frame + 1;
    else
        steps_out_of_frame = 0;
    end
    
    % Terminate the episode if the point has been out of frame for too many consecutive steps
    if steps_out_of_frame >= max_consecutive_steps
        isTerminal = true;
    else
        isTerminal = false;
    end

    % Terminate the episode when the maximum number of episodes is reached
    if episodeCount >= max_episodes
        isTerminal = true;
    else
        isTerminal = false;
    end
end


%% QUADCOPTER DYNAMICS
function new_state = quadcopter_dynamics(current_state, action, mass, dt)
    % Define constants
    g = 9.81;  % gravitational acceleration (m/s^2)

    % Unpack current state
    position = current_state(1:3);
    velocity = current_state(4:6);
    % current_roll = current_state(7);
    % current_pitch = current_state(8);
    % current_yaw = current_state(9);

    roll = action(1);
    pitch = action(2);
    yaw = action(3);

    % Compute acceleration due to thrust and gravity
    R = rotationMatrixEulerZYX(roll, pitch, yaw);
    thrust_force = [0; 0; thrust];
    gravity_force = [0; 0; -g];
    total_force = R * thrust_force + gravity_force;
    acceleration = total_force / mass;  % m is the mass of the quadcopter

    % Update velocity and position
    new_velocity = velocity + acceleration * dt;
    new_position = position + new_velocity * dt;

    % Update roll, pitch, and yaw angles
    new_roll = roll;
    new_pitch = pitch;
    new_yaw = yaw;

    % Update the state vector
    new_state = [new_position; new_velocity; new_roll; new_pitch; new_yaw];
end

function R = rotationMatrixEulerZYX(roll, pitch, yaw)
% Euler ZYX angles convention
    Rx = [ 1,           0,          0;
           0,           cos(roll),  -sin(roll);
           0,           sin(roll),   cos(roll) ];
    Ry = [ cos(pitch),  0,          sin(pitch);
           0,           1,          0;
          -sin(pitch),  0,          cos(pitch) ];
    Rz = [cos(yaw),    -sin(yaw),   0;
          sin(yaw),     cos(yaw),   0;
          0,            0,          1 ];
    if nargout == 3
        % Return rotation matrix per axes
        return;
    end
    % Return rotation matrix from body frame to inertial frame
    R = Rz*Ry*Rx;
end

function R = rotationMatrix(angles)
    roll = angles(1);
    pitch = angles(2);
    yaw = angles(3);
    
    R_x = [1, 0, 0;
           0, cos(roll), -sin(roll);
           0, sin(roll), cos(roll)];
    
    R_y = [cos(pitch), 0, sin(pitch);
           0, 1, 0;
           -sin(pitch), 0, cos(pitch)];
    
    R_z = [cos(yaw), -sin(yaw), 0;
           sin(yaw), cos(yaw), 0;
           0, 0, 1];
    
    R = R_z * R_y * R_x;
end

% Function to check if a point is within the camera's FOV
function withinFOV = isWithinFOV(point, fov_horizontal, fov_vertical)
    if abs(atan2(point(2), point(1))) <= fov_horizontal/2 && abs(atan2(point(3), point(1))) <= fov_vertical/2
        withinFOV = true;
    else
        withinFOV = false;
    end
end






