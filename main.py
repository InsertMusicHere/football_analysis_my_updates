from utils import read_video, save_video,measure_distance
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator
import streamlit as st
from development_and_analysis import is_point_in_polygon,is_near_goal_post
import pandas as pd

def stat_func(count_dict):

    player_goal_counts = {}

    player_numbers = count_dict["player_number"]
    goal_posts = count_dict["goal_post"]

    for player, goal_post in zip(player_numbers, goal_posts):
        if player not in player_goal_counts:
            player_goal_counts[player] = {"left": 0, "right": 0}
        player_goal_counts[player][goal_post] += 1

    # Create the results string
    results = []
    for player, goals in player_goal_counts.items():
        if goals["left"] > 0:
            results.append(f"Player {player} was near the left goal's penalty area {goals['left']} time(s).")
        if goals["right"] > 0:
            results.append(f"Player {player} was near the right goal's penalty area {goals['right']} time(s).")

    return results

# def main():

url_ = st.text_input("Please add a video URL or Keep this field blank to use default video",disabled=True)

if url_:
    video_file = open(url_, "rb")
    video_frames = video_file.read()
else:

    video_file = open('input_videos/left_area_goal.mp4', "rb")
    video_frames = video_file.read()

st.video(video_frames)

start_ = st.button("Start ML Process")

final = 0

count_dict = {
                "player_number": [],
                "goal_post": []
            }

if start_:

    with st.spinner("'Kicking off analysis: Dribbling through the data...'"):


        # Read Video
        video_frames = read_video('input_videos/left_area_goal.mp4')


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

        penalty_area_coordinates = {'right_penalty_area': [(1365, 220), (1026, 250), (1458, 490), (1894, 500)],
            'left_penalty_area': [(9, 540), (314, 676), (791, 317), (428, 298)]}

        left_penalty_area = penalty_area_coordinates["left_penalty_area"]
        right_penalty_area = penalty_area_coordinates["right_penalty_area"]

        


        # Assign Ball Acquisition
        player_assigner = PlayerBallAssigner()
        team_ball_control = []

        for frame_num, player_track in enumerate(tracks['players']):
            for player_id, player in player_track.items():
                if 'has_ball' not in player:
                    player['has_ball'] = False

            try:
                ball_bbox = tracks['ball'][frame_num][1]['bbox']
                assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
            except KeyError:
                assigned_player = -1

            if assigned_player != -1:
                tracks['players'][frame_num][assigned_player]['has_ball'] = True
                team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

                # Check if the player is near a goal post
                player_position = player_track[assigned_player]['position_adjusted']
                near_goal, goal_post = is_near_goal_post(player_position, left_penalty_area, right_penalty_area)

                


                if near_goal:
                    text = f"Player {assigned_player} in possession of the ball, is near the {goal_post} goal post of the penalty box"
                    count_dict["player_number"].append(assigned_player)
                    count_dict["goal_post"].append(goal_post)

                    # Draw the text on the frame inside a rectanlge
                    # parameters include the frame, the text to be displayed, the position (10 pixels from the left and 30 pixels from the bottom of the frame), the font, font scale, color (white), thickness, and line type.
                    # cv2.rectangle(video_frames, (90, 910), (430,1000), (255,255,255), -1 )
                        
                    video_frames[frame_num] = cv2.putText(
                        video_frames[frame_num], text, (10, video_frames[frame_num].shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA
                    )

                else:

                    video_frames[frame_num] = cv2.putText(
                        video_frames[frame_num], "At the moment, no player with the ball is close to the penalty area", (10, video_frames[frame_num].shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                    )

            else:

                video_frames[frame_num] = cv2.putText(
                    video_frames[frame_num], "At the moment, no player with the ball is close to the penalty area", (10, video_frames[frame_num].shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA
                )

        team_ball_control= np.array(team_ball_control)


        # Draw output 
        ## Draw object Tracks
        output_video_frames = tracker.draw_annotations(video_frames, tracks,team_ball_control)

        ## Draw Camera movement
        output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)

        ## Draw Speed and Distance
        speed_and_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

        # Save video
        save_video(output_video_frames, 'output_videos/left_area_goal_output.mp4')

        final = 1

        
        

if final == 1:

    video_file = open('output_videos/left_area_goal_output.mp4', "rb")
    video_frames = video_file.read()

    st.video(video_frames)

    # results = stat_func(count_dict)
    # for result in results:
    #     st.write(result)

    player_numbers = count_dict["player_number"]
    goal_posts = count_dict["goal_post"]

    df_ = pd.DataFrame(count_dict)

    df_["Number of times near the Goal post"] = df_.groupby("player_number")["goal_post"].transform("count")

    df_ = df_.drop_duplicates()
    st.info("Potential Statistics that can be done in future with this project")
    st.write(df_)









# if __name__ == '__main__':
#     main()
