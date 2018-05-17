package com.emaraic.utils;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.font.FontRenderContext;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import javax.imageio.ImageIO;

/**
 *Kindly, Don't Remove this Header.
 * This class generates digits dataset with all fonts included in the OS with
 * different sizes from 20 to 100, Some of these fonts include symbols not digits 
 * or characters so I excluded them only for mac and windows, for linux users
 * must check for that.
 *
 * @author Taha Emara 
 * Website: http://www.emaraic.com 
 * Email : taha@emaraic.com
 * Created on: May 10, 2018
 */
public class GenerateDataset {

    private final static String DATA[] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"};
    private final static String PARENT_PATH = "data";

    public static BufferedImage resize(BufferedImage img, int newW, int newH) {
        Image tmp = img.getScaledInstance(newW, newH, Image.SCALE_SMOOTH);
        BufferedImage dimg = new BufferedImage(newW, newH, BufferedImage.TYPE_BYTE_GRAY);
        Graphics2D g2d = dimg.createGraphics();
        g2d.drawImage(tmp, 0, 0, null);
        g2d.dispose();
        return dimg;
    }

    public static void main(String[] args) {

        /*Exclude symobol fonts from generating images*/
        final String EXCEPTIONS[] = {"Webdings", "Bodoni Ornaments", "Wingdings", "Wingdings 2", "Wingdings 3",//exclude these fonts for mac users
            "Symbol", "Marlett", "HoloLens MDL2 Assets", "Segoe MDL2 Assets", "Gabriola"};//exclude these fonts for windows users
        String FONTS[] = GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames();
        List<String> fonts = new ArrayList<>(Arrays.asList(FONTS));
        for (String exception : EXCEPTIONS) {
            fonts.remove(exception);
        }
        FONTS = fonts.toArray(new String[0]);

        Font font = new Font("Arial", Font.PLAIN, 48);
        BufferedImage img = new BufferedImage(1, 1, BufferedImage.TYPE_BYTE_GRAY);

        for (String data : DATA) {
            String path = PARENT_PATH + File.separator + data;
            new File(path).mkdirs();

            for (String fnt : FONTS) {
                for (int size = 20; size < 100; size++) {
                    font = new Font(fnt, Font.PLAIN, size);

                    FontRenderContext frc = new FontRenderContext(null, true, true);

                    //get the height and width of the text
                    Rectangle2D bounds = font.getStringBounds(data, frc);
                    int width = (int) bounds.getWidth() + 2;
                    int height = (int) bounds.getHeight() + 2;

                    img = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
                    Graphics2D g2d = img.createGraphics();
                    g2d.setFont(font);

                    g2d.setRenderingHint(RenderingHints.KEY_ALPHA_INTERPOLATION, RenderingHints.VALUE_ALPHA_INTERPOLATION_QUALITY);
                    g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    g2d.setRenderingHint(RenderingHints.KEY_COLOR_RENDERING, RenderingHints.VALUE_COLOR_RENDER_QUALITY);
                    g2d.setRenderingHint(RenderingHints.KEY_DITHERING, RenderingHints.VALUE_DITHER_ENABLE);
                    g2d.setRenderingHint(RenderingHints.KEY_FRACTIONALMETRICS, RenderingHints.VALUE_FRACTIONALMETRICS_ON);
                    g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
                    g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
                    g2d.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);
                    g2d.setBackground(Color.BLACK);
                    g2d.setColor(Color.WHITE);
                    g2d.drawString(data, (float) bounds.getX(), (float) -bounds.getY());
                    g2d.dispose();
                    img = resize(img, 28, 28);
                    try {
                        ImageIO.write(img, "png", new File(path + File.separator
                                + data + "_" + size + "_" + fnt + ".png"));
                    } catch (IOException ex) {
                        ex.printStackTrace();
                    }
                }
            }
        }
    }
}
