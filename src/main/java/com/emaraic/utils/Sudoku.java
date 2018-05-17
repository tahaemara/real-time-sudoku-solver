package com.emaraic.utils;

import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * This code is based on this code
 * https://github.com/bob-carpenter/java-sudoku/blob/master/Sudoku.java
 * and this http://www.cs.armstrong.edu/liang/apcs/html/CheckSudokuSolutionWithLineNumber.html
*
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: Apr 14, 2018
 */
public class Sudoku {

    public static boolean isValid(double[] input) {
        double[][] grid = new double[9][9];
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                grid[i][j] = input[(i * 9) + j];
                //System.out.print( grid[i][j] +" , ");
            }
            //System.out.println("");
        }
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if(grid[i][j]==0)continue;
                if (grid[i][j] < 1 || grid[i][j] > 9
                        || !isValid(i, j, grid)) {
                    return false;
                }
            }
        }
        return true; // The solution is valid
    }

    public static boolean isValid(int i, int j, double[][] grid) {
        // Check whether grid[i][j] is valid at the i's row
        for (int column = 0; column < 9; column++) {
            if (column != j && grid[i][column] == grid[i][j]) {
                return false;
            }
        }

        // Check whether grid[i][j] is valid at the j's column
        for (int row = 0; row < 9; row++) {
            if (row != i && grid[row][j] == grid[i][j]) {
                return false;
            }
        }

        // Check whether grid[i][j] is valid in the 3 by 3 box
        for (int row = (i / 3) * 3; row < (i / 3) * 3 + 3; row++) {
            for (int col = (j / 3) * 3; col < (j / 3) * 3 + 3; col++) {
                if (row != i && col != j && grid[row][col] == grid[i][j]) {
                    return false;
                }
            }
        }
        return true; // The current value at grid[i][j] is valid
    }

    public static boolean solve(int i, int j, INDArray cells) {
        if (i == 9) {
            i = 0;
            if (++j == 9) {
                return true;
            }
        }
        if (cells.getDouble(i, j) != 0) // skip filled cells
        {
            return solve(i + 1, j, cells);
        }

        for (int val = 1; val <= 9; ++val) {
            if (legal(i, j, val, cells)) {
                cells.putScalar(i, j, val);
                if (solve(i + 1, j, cells)) {
                    return true;
                }
            }
        }
        cells.putScalar(i, j, 0); // reset on backtrack
        return false;
    }

    private static boolean legal(int i, int j, int val, INDArray cells) {
        for (int k = 0; k < 9; ++k) // row
        {
            if (val == cells.getDouble(k, j)) {
                return false;
            }
        }

        for (int k = 0; k < 9; ++k) // col
        {
            if (val == cells.getDouble(i, k)) {
                return false;
            }
        }

        int boxRowOffset = (i / 3) * 3;
        int boxColOffset = (j / 3) * 3;
        for (int k = 0; k < 3; ++k) // box
        {
            for (int m = 0; m < 3; ++m) {
                if (val == cells.getDouble(boxRowOffset + k, boxColOffset + m)) {
                    return false;
                }
            }
        }

        return true;
    }

}
