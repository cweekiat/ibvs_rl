function new_state = quadcopter_dynamics(current_state, action, mass, dt)
    % Define constants
    g = 9.81;  % gravitational acceleration (m/s^2)

    % Unpack current state
    position = current_state(1:3);
    velocity = current_state(4:6);
    current_roll = current_state(7);
    current_pitch = current_state(8);
    current_yaw = current_state(9);

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