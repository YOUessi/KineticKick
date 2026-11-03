"""Test KineticKick goal-relative geometry without Isaac Sim/GPU."""
import math,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from goal_conditioned_planner import goal_conditioned_plan, sub, left, length

class TestGoalPlanning(unittest.TestCase):
    def test_shot_points_to_goal_and_support_is_left(self):
        p=goal_conditioned_plan(ball_xy=(1.,2.),goal_xy=(4.,2.),
             robot_xy=(-1.,2.),surface="instep_top")
        self.assertEqual(p["shot_direction_world"],[1.,0.])
        self.assertAlmostEqual(p["shot_heading_rad"],0.)
        self.assertEqual(p["support_foot"],"left")
        self.assertGreater(p["support_foot_plant_xy"][1],p["ball_world_xy"][1])
        self.assertLess(p["approach_xy"][0],p["ball_world_xy"][0])
        self.assertGreater(p["follow_through_xy"][0],p["ball_world_xy"][0])

    def test_rotational_equivariance_90_deg(self):
        p=goal_conditioned_plan(ball_xy=(0.,0.),goal_xy=(3.,0.),robot_xy=(-1.,0.))
        q=goal_conditioned_plan(ball_xy=(0.,0.),goal_xy=(0.,3.),robot_xy=(0.,-1.))
        # Planner JSON rounds heading to 6 decimal places.
        self.assertAlmostEqual(q["shot_heading_rad"],math.pi/2,places=6)
        for key in ("approach_xy","support_foot_plant_xy","swing_foot_start_xy",
                    "impact_target_xy","follow_through_xy","preplant_base_xy"):
            a=p[key];b=q[key]
            self.assertAlmostEqual(-a[1],b[0],places=6)
            self.assertAlmostEqual(a[0],b[1],places=6)

    def test_left_foot_mirrors_support_side(self):
        r=goal_conditioned_plan(ball_xy=(0.,0.),goal_xy=(3.,0.),
              robot_xy=(-1.,0.),foot="right")
        l=goal_conditioned_plan(ball_xy=(0.,0.),goal_xy=(3.,0.),
              robot_xy=(-1.,0.),foot="left")
        self.assertEqual(l["support_foot"],"right")
        self.assertAlmostEqual(l["support_foot_plant_xy"][1],
                               -r["support_foot_plant_xy"][1])

    def test_relative_planner_does_not_depend_on_absolute_position(self):
        a=goal_conditioned_plan(ball_xy=(0.,0.),goal_xy=(3.,.5),
               robot_xy=(-1.,0.),surface="medial")
        b=goal_conditioned_plan(ball_xy=(10.,-7.),goal_xy=(13.,-6.5),
               robot_xy=(9.,-7.),surface="medial")
        self.assertEqual(a["shot_heading_deg"],b["shot_heading_deg"])
        for key in ("approach_xy","preplant_base_xy","support_foot_plant_xy"):
            self.assertAlmostEqual(b[key][0]-a[key][0],10.)
            self.assertAlmostEqual(b[key][1]-a[key][1],-7.)

    def test_invalid_plan_rejected(self):
        common=dict(robot_xy=(0.,0.),ball_xy=(0.,0.),goal_xy=(3.,0.))
        with self.assertRaises(ValueError):
            goal_conditioned_plan(**{**common,"goal_xy":(0.,0.)})
        with self.assertRaises(ValueError):
            goal_conditioned_plan(**{**common,"foot":"both"})
        with self.assertRaises(ValueError):
            goal_conditioned_plan(**{**common,"target_ball_speed_mps":900})
        with self.assertRaises(ValueError):
            goal_conditioned_plan(**{**common,"surface":"toe"})

if __name__=="__main__":unittest.main()
