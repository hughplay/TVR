import Two from 'two.js';
import SvgSaver from 'svgsaver';

var Diagram = {
    update_diagram: function(div_id, objs ,scale, illustrator) {
      var elem = document.getElementById(div_id);
      if (elem === null) {
        return;
      }
      elem.innerHTML = '';
      elem.style.cssText = "font-family: Palatino, serif;"
      var sizes = { small: 1.5, medium: 3, large: 6 };
      var max_points = 40;
      var bleed = sizes.large;
      var n_points = (max_points + bleed) * 2 + 1;
      var offset = (max_points + bleed) * scale;
      var area = max_points * scale;
      var len = scale * n_points;
      var params = { width: len, height: len };

      var scene = new Two(params).appendTo(elem);

      var to_plot = function(x, y) {
        return [y * scale + offset, x * scale + offset];
      };

      // Plot Coodinate
      var invisible_area = scene.makeRectangle(
        offset,
        offset,
        area * 2,
        area * 2
      );
      var visible_area = scene.makeRectangle(offset, offset, area, area);
      invisible_area.fill = '#f5f5f5';
      visible_area.fill = 'white';
      visible_area.noStroke();
      invisible_area.noStroke();

      var line_gap = 10;
      var n_line = (max_points / line_gap) * 2 + 1;
      for (var i = 0; i < n_line; i++) {
        var pos_1 = to_plot(-40, -40 + i * line_gap);
        var pos_2 = to_plot(40, -40 + i * line_gap);
        var line_h = scene.makeLine(pos_1[1], pos_1[0], pos_2[1], pos_2[0]);
        var line_v = scene.makeLine(pos_1[0], pos_1[1], pos_2[0], pos_2[1]);
        line_h.linewidth = line_v.linewidth = scale * 0.05;
        line_h.stroke = line_v.stroke = '#aaa';
      }

      var invisible_area_line = scene.makeRectangle(
        offset,
        offset,
        area * 2,
        area * 2
      );
      var visible_area_line = scene.makeRectangle(offset, offset, area, area);
      visible_area_line.linewidth = invisible_area_line.linewidth = scale * 0.10;
      visible_area_line.stroke = invisible_area_line.storke = 'black';
      visible_area_line.fill = invisible_area_line.fill = 'none';

      // Plot signs
      var pos_zero = to_plot(0, 0);
      var pos_x_axis = to_plot(10, 0);
      var pos_y_axis = to_plot(0, 10);
      scene.makeLine(pos_zero[0], pos_zero[1], pos_x_axis[0], pos_x_axis[1]);
      scene.makeLine(pos_zero[0], pos_zero[1], pos_y_axis[0], pos_y_axis[1]);
      var arrow_x = scene.makePolygon(pos_x_axis[0], pos_x_axis[1], scale);
      var arrow_y = scene.makePolygon(pos_y_axis[0], pos_y_axis[1], scale);
      arrow_x.rotation = Math.PI;
      arrow_y.rotation = Math.PI / 2;
      arrow_x.fill = arrow_y.fill = 'black';
      scene.makeText(
        'x',
        pos_x_axis[0] + 2 * scale,
        pos_x_axis[1] + 2 * scale,
        {
          size: scale * 4,
          weight: 800,
          family: 'Palatino, serif',
          style: 'italic',
        }
      );
      scene.makeText(
        'y',
        pos_y_axis[0] + 2 * scale,
        pos_y_axis[1] - 2 * scale,
        {
          size: scale * 4,
          weight: 800,
          family: 'Palatino, serif',
          style: 'italic',
        }
      );

      var pos_right_start = to_plot(-35, -35);
      var pos_right_end = to_plot(-35, -25);
      var right_line = scene.makeLine(
        pos_right_start[0], pos_right_start[1],
        pos_right_end[0], pos_right_end[1]);
      var pos_behind_start = to_plot(-35, 35);
      var pos_behind_end = to_plot(-25, 35);
      var behind_line = scene.makeLine(
        pos_behind_start[0], pos_behind_start[1],
        pos_behind_end[0], pos_behind_end[1]);
      var arrow_right = scene.makePolygon(
        pos_right_end[0], pos_right_end[1], scale);
      var arrow_behind = scene.makePolygon(
        pos_behind_end[0], pos_behind_end[1], scale);
      arrow_right.rotation = Math.PI / 2;
      arrow_behind.rotation = Math.PI;
      var pos_right = to_plot(-30, -32);
      var text_right = scene.makeText(
        'right',
        pos_right[0] + 2 * scale,
        pos_right[1] - 2 * scale,
        {
          size: scale * 3.5,
          weight: 200,
        }
      );
      var pos_behind = to_plot(-28, 26);
      var text_behind = scene.makeText(
        'behind',
        pos_behind[0] + 2 * scale,
        pos_behind[1] - 2 * scale,
        {
          size: scale * 3.5,
          weight: 200,
        }
      );
      text_behind.fill = text_right.fill = arrow_right.stroke = arrow_behind.stroke = arrow_right.fill = arrow_behind.fill = right_line.stroke = behind_line.stroke = '#ccc';
      behind_line.linewidth = right_line.linewidth = scale * 0.1;
      text_behind.family = text_right.family = 'Palatino, serif';
      text_behind.style = text_right.style = 'italic';
      text_behind.weight = text_right.weight = 'bold';

      var pos_vis = to_plot(-12, 5);
      var text_vis = scene.makeText(
        'visible area',
        pos_vis[0] + 2 * scale,
        pos_vis[1] - 2 * scale,
        {
          size: scale * 3.5,
          weight: 800,
        }
      );
      var pos_invis = to_plot(30, -20);
      var text_invis = scene.makeText(
        'invisible area',
        pos_invis[0] + 2 * scale,
        pos_invis[1] - 2 * scale,
        {
          size: scale * 3.5,
          weight: 800,
        }
      );
      text_vis.fill = text_invis.fill = '#ccc';
      text_vis.style = text_invis.style = 'italic';
      text_vis.family = text_invis.family = 'Palatino, serif';

      // Plot objects
      for (let idx in objs) {
        var obj = objs[idx];
        var size = sizes[obj.size] * scale;
        var x = obj.position[0];
        var y = obj.position[1];
        var plot_pos = to_plot(x, y);
        var plot;
        if (obj.shape == 'cube') {
          plot = scene.makeRectangle(
            plot_pos[0],
            plot_pos[1],
            size * Math.sqrt(2),
            size * Math.sqrt(2)
          );
        } else if (obj.shape == 'sphere') {
          plot = scene.makeCircle(plot_pos[0], plot_pos[1], size);
        } else if (obj.shape == 'cylinder') {
          plot = scene.makeCircle(plot_pos[0], plot_pos[1], size);
          if (illustrator) {
            plot.dashes = [size * 0.1, size * 0.1];
          } else {
            plot.dashes = [size * 0.3, size * 0.3];
          }
        } else {
          console.log('Wrong shape ' + obj.shape);
        }
        plot.fill = obj.color;
        if (illustrator) {
          plot_pos[1] += 1. * scale;
        }
        scene.makeText(idx, plot_pos[0], plot_pos[1], {
          size: obj.size == 'small' ? scale * 2.5 : scale * 4,
          weight: 800,
          fill: obj.color == 'yellow' || obj.color == 'cyan' ? '#777' : '#eee',
          family: 'Palatino, serif'
        });

        var sign = '';
        if (obj.material == 'glass') {
          sign = 'G';
        } else if (obj.material == 'metal') {
          sign = 'M';
        } else if (obj.material == 'rubber') {
          sign = 'R';
        }
        var mat = scene.makeText(
          sign,
          plot_pos[0],
          x < -max_points * 0.75
            ? plot_pos[1] + size + 2.5 * scale
            : plot_pos[1] - size - 2 * scale,
          {
            size: scale * 3,
            weight: 800,
            fill: '#777',
            family: 'Palatino, serif'
          }
        );

        if (x >= -20 && x <= 20 && y >= -20 && y <= 20) {
          mat.opacity = plot.opacity = 0.8;
        } else {
          mat.opacity = plot.opacity = 0.3;
        }

        if (illustrator) {
          plot.linewidth = scale * 0.2;
        } else {
          plot.linewidth = scale * 0.5;
        }
        plot.stroke = '#333';
        plot.rotation = -obj.rotation;
      }

      scene.update();
    },

    save_diagram: function(div_id, filetype) {
      var saver = new SvgSaver();
      var ele = document.getElementById(div_id).children[0];
      if (filetype == 'png') {
        saver.asPng(ele, 'diagram.png');
      } else if (filetype == 'svg') {
        saver.asSvg(ele, 'diagram.svg');
      }
    }
}

export { Diagram }
