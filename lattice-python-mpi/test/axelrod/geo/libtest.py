#  Copyright (C) 2011 Jens Pfau <jpfau@unimelb.edu.au>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import unittest

import sys
sys.path.append("../../../src/axelrod/geo/")
import lib

from numpy import array


class MylibTest(unittest.TestCase):

    def testCulturesIn(self):
        C = [array([3,4]), array([3,3]), array([3,3]), array([3,4])]
        self.assertEqual(2, lib.cultures_in(C, [2,3])[1])
        self.assertEqual(2, lib.cultures_in(C, [0,1,2,3])[1])
        self.assertEqual(1, lib.cultures_in(C, [0])[1])
        
        self.assertEqual([0,1,1,0], lib.cultures_in(C, [0,1,2,3])[0])

        
      

    def testCalcDiversityOfCluster(self):
        C = [array([3,4]), array([3,3]), array([3,3]), array([3,4]), array([4,5])]
        
        cd1 = lib.calcDiversityOfCluster(C, [1,2])
        cd2 = lib.calcDiversityOfCluster(C, [0,1])
        cd3 = lib.calcDiversityOfCluster(C, [3,4])
        
        self.assertTrue(cd1 < cd2)
        self.assertTrue(cd2 < cd3)
        self.assertAlmostEqual(0.0, cd1)
        
        
    def testCalcModal(self):
        F = 2
        q = 6
        
        C = [array([3,4]), array([3,3]), array([3,3]), array([3,4]), array([4,4])]
        
        m1 = lib.calcModal(C, [1,2], F, q)
        m2 = lib.calcModal(C, [0,1], F, q)
        m3 = lib.calcModal(C, [0,1,2], F, q)
        m4 = lib.calcModal(C, [2,3,4], F, q)
        
        self.assertEqual(3, m1[0])
        self.assertEqual(3, m1[1])
        self.assertEqual(3, m2[0])
        self.assertEqual(3, m2[1])
        self.assertEqual(3, m3[0])
        self.assertEqual(3, m3[1])
        self.assertEqual(3, m4[0])
        self.assertEqual(4, m4[1])
        
        
    def testCalcDiversity(self):
        F = 2
        q = 6        
        
        C1 = [array([3,4]), array([3,3]), array([3,3]), array([3,4]), array([4,5])]
        cluster1 = [[1,2], [0,3,4]]
        C2 = [array([3,4]), array([3,3]), array([3,3]), array([3,4]), array([4,5])]
        cluster2 = [[0,1], [2,3,4]]
        C3 = [array([3,4]), array([1,2]), array([1,3]), array([5,4]), array([4,5])]
        cluster3 = [[0,1], [2,3,4]]
        C4 = [array([3,4]), array([1,2]), array([1,3]), array([5,4]), array([4,5])]
        cluster4 = [[0], [1], [2], [3], [4]]   
        
        wcd1, bcd1, ratio1, size_x_div1 = lib.calcDiversity(C1, cluster1, F, q)
        wcd2, bcd2, ratio2, size_x_div2 = lib.calcDiversity(C2, cluster2, F, q)
        wcd3, bcd3, ratio3, size_x_div3 = lib.calcDiversity(C3, cluster3, F, q)
        wcd4, bcd4, ratio4, size_x_div4 = lib.calcDiversity(C4, cluster4, F, q)
        
        self.assertTrue(wcd1 < wcd2)
        self.assertTrue(bcd1 > bcd2)
        self.assertTrue(wcd2 < wcd3)        
        self.assertTrue(bcd1 == bcd3)
        self.assertTrue(ratio1 > ratio2)
        self.assertTrue(ratio2 < ratio3)
        self.assertEqual(0.0, ratio2)
        self.assertEqual(0.0, wcd4)
        self.assertTrue(float('nan') != ratio4)
        
        
        self.assertEqual(2, size_x_div1[0][0])
        self.assertEqual(3, size_x_div1[1][0])
        self.assertTrue(size_x_div1[0][1] < size_x_div1[1][1])        

        self.assertEqual(2, size_x_div2[0][0])
        self.assertEqual(3, size_x_div2[1][0]) 
        self.assertTrue(size_x_div2[0][1] < size_x_div2[1][1])       

        self.assertEqual(2, size_x_div3[0][0])
        self.assertEqual(3, size_x_div3[1][0])  
        self.assertTrue(size_x_div3[0][1] == size_x_div3[1][1])     
        
        self.assertEqual(1, size_x_div4[0][0])
        self.assertEqual(1, size_x_div4[1][0])   
        self.assertEqual(0, size_x_div4[0][1])
        self.assertEqual(0, size_x_div4[1][1])     
        
        

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()