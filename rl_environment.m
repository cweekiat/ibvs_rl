% Define custom state transition function (modify as needed)
function nextState = customStateTransition(camera_state, quad_state, action)
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