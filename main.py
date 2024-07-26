from utils import read_video, save_video,measure_distance
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator


def main():
    
    # Read Video
    video_frames = read_video('input_videos/A1606b0e6_0.mp4')

    # Initialize Tracker
    tracker = Tracker('models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl',
                                       stub_path_detaction='stubs/detection_stubs.pkl')
    # Get object positions 
    tracker.add_position_to_tracks(tracks)

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)


    # View Trasnformer
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], 
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # Define goal post positions (assuming fixed positions for simplicity)
    goal_post_positions = {
        "left": (50, 300),  # example coordinates for left goal post
        "right": (950, 300) # example coordinates for right goal post
    }
    goal_post_threshold = 200  # threshold distance to consider a player near the goal post

    def is_near_goal_post(player_position, goal_post_positions, threshold):
        for post, position in goal_post_positions.items():
            distance = measure_distance(player_position, position)
            # print(f"Distance from player to {post} goal post: {distance}") 
            if distance < threshold:
                return True, post
        return False, None
    
    # Assign Ball Aquisition
    player_assigner =PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        
        print(print(tracks['ball'][frame_num]))
        try:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        except:
            pass
        
        try:
            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

                # Check if the player is near a goal post
                player_position = player_track[assigned_player]['position_adjusted']
                # print(f"Frame {frame_num}: Player {assigned_player} position: {player_position}")
                near_goal, goal_post = is_near_goal_post(player_position, goal_post_positions, goal_post_threshold)
                print("near goal",near_goal, goal_post)

                if near_goal:
                    text = f"Player {assigned_player} near the {goal_post} goal post."
                    print(text)

                    # Draw the text on the frame
                    #  parameters include the frame, the text to be displayed, the position (10 pixels from the left and 30 pixels from the bottom of the frame), the font, font scale, color (white), thickness, and line type.
                    video_frames[frame_num] = cv2.putText(
                        video_frames[frame_num], text, (10, video_frames[frame_num].shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )
                    
        except:
            pass


        # else:
        #     try:
        #         team_ball_control.append(team_ball_control[-1])
                
        #     except:
        #         team_ball_control.append(team_ball_control[0])
    team_ball_control= np.array(team_ball_control)


    # Draw output 
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    # Save video
    save_video(output_video_frames, 'output_videos/output_video2nd.avi')

if __name__ == '__main__':
    main()