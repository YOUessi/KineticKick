"""Procedural visual soccer ball geometry regression tests (no Isaac Sim)."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from classic_soccer_visual import soccer_panel_polygons,triangle_spherical_panels

class SoccerPanelTests(unittest.TestCase):
    def test_classic_soccer_counts_and_topology(self):
        pent,hexagon=soccer_panel_polygons()
        self.assertEqual((len(pent),len(hexagon)),(12,20))
        self.assertTrue(all(len(p)==5 for p in pent))
        self.assertTrue(all(len(h)==6 for h in hexagon))
        self.assertEqual(sum(len(x) for x in pent+hexagon),180)

    def test_panels_lie_on_soccer_ball_shell(self):
        pent,hexagon=soccer_panel_polygons()
        for typ,polygons in (("pent",pent),("hex",hexagon)):
            pts,norm,cnt,idx=triangle_spherical_panels(polygons)
            lengths=np.linalg.norm(np.array(pts),axis=1)
            nlen=np.linalg.norm(np.array(norm),axis=1)
            self.assertTrue(np.allclose(lengths,.1105,atol=1e-6),typ)
            self.assertTrue(np.allclose(nlen,1.,atol=1e-6),typ)
            self.assertEqual(3*len(cnt),len(idx))
            self.assertEqual(len(pts),len(norm))
            self.assertLess(max(idx),len(pts))

    def test_panels_are_not_another_physics_ball(self):
        # No PhysicsRigidBody or CollisionAPI anywhere in numeric generator;
        # only attach_classic_soccer_visual() creates render-only leaf Meshes.
        import inspect
        from classic_soccer_visual import attach_classic_soccer_visual
        source=inspect.getsource(attach_classic_soccer_visual)
        self.assertIn("UsdGeom.Mesh.Define",source)
        self.assertNotIn("UsdPhysics.CollisionAPI.Apply",source)
        self.assertNotIn("RigidObject(",source)

if __name__=="__main__":
    unittest.main()
