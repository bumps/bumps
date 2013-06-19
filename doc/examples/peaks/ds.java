import java.util.*;
import javax.swing.*;
import java.io.*;
import java.awt.*;
import java.awt.image.*;

public class ds extends Canvas {
	public static void main(String args[]) {
		JFrame window = new JFrame();
		ds c = new ds();
		
		window.setSize(c.datawidth+16,c.dataheight+38);
		window.add(c);
		window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		window.setVisible(true);
	}
	
	private double[][] vals;
	public int datawidth,dataheight;
	private double min,max;
	public ds() {
		super();
		try {
			Scanner fin = new Scanner(new File("XY_mesh2.txt"));
			ArrayList<double[]> data = new ArrayList<double[]>();
			String line;
			String[] parts;
			double[] cols;
			min = Integer.MAX_VALUE;
			max = Integer.MIN_VALUE;
			while (fin.hasNextLine()) {
				System.out.println(dataheight);
				line = fin.nextLine();
				parts = line.trim().replaceAll(" +"," ").split(" ");
				cols = new double[parts.length];
				for (int i=0; i<parts.length; i++) {
					cols[i] = Double.parseDouble(parts[i]);
					if (cols[i]<min) min = cols[i];
					if (cols[i]>max) max = cols[i];
				}
				datawidth = cols.length;
				data.add(cols);
				dataheight++;
			}
			vals = new double[dataheight][datawidth];
			
			for (int i=0; i<dataheight; i++) {
				for (int j=0; j<datawidth; j++) {
					vals[i][j] = data.get(i)[j];
				}
			}
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void paint(Graphics g) {
		int[] pixels = new int[datawidth*dataheight];
		int bri = 0;
		double diff = max-min;
		int sc = 5;
		
		for (int i=0; i<dataheight; i++) {
			for (int j=0; j<datawidth; j++) {
				bri = (int) ((vals[i][j]-min)/diff*255) & 0xFF;
				pixels[i*dataheight+j] = (0xFF<<24) | (bri<<16) | (bri<<8) | bri;
				g.setColor(new Color(pixels[i*dataheight+j]));
				g.fillRect(i*sc,j*sc,sc,sc);
			}
			System.out.println();
		};
	}
}